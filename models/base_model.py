import os
import torch
from collections import OrderedDict
from . import networks


class BaseModel():

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.auxiliary_dir = os.path.join(opt.checkpoints_dir, opt.auxiliary_root)
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # load and print networks; create schedulers
    def setup(self, opt, parser=None):
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    # make models eval mode during test time
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # save models to the disk
    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
    
    # save generators to one file and discriminators to another file
    def save_networks2(self, which_epoch):
        gen_name = os.path.join(self.save_dir, '%s_net_gen.pt' % (which_epoch))
        dis_name = os.path.join(self.save_dir, '%s_net_dis.pt' % (which_epoch))
        dict_gen = {}
        dict_dis = {}
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    state_dict = net.module.cpu().state_dict()
                    net.cuda(self.gpu_ids[0])
                else:
                    state_dict = net.cpu().state_dict()
                
                if name[0] == 'G':
                    dict_gen[name] = state_dict
                elif name[0] == 'D':
                    dict_dis[name] = state_dict
        torch.save(dict_gen, gen_name)
        torch.save(dict_dis, dis_name)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def load_networks(self, which_epoch):
        gen_name = os.path.join(self.save_dir, '%s_net_gen.pt' % (which_epoch))
        if os.path.exists(gen_name):
            self.load_networks2(which_epoch)
            return
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
    
    def load_networks2(self, which_epoch):
        gen_name = os.path.join(self.save_dir, '%s_net_gen.pt' % (which_epoch))
        gen_state_dict = torch.load(gen_name, map_location=str(self.device))
        if self.isTrain:
            dis_name = os.path.join(self.save_dir, '%s_net_dis.pt' % (which_epoch))
            dis_state_dict = torch.load(dis_name, map_location=str(self.device))
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                if name[0] == 'G':
                    print('loading the model from %s' % gen_name)
                    state_dict = gen_state_dict[name]
                elif name[0] == 'D':
                    print('loading the model from %s' % dis_name)
                    state_dict = dis_state_dict[name]
                
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
    
    # load auxiliary net models from the disk
    def load_auxiliary_networks(self):
        for name in self.auxiliary_model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % ('latest', name)
                load_path = os.path.join(self.auxiliary_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # =============================================================================================================
    def inverse_mask(self, mask):
        return torch.ones(mask.shape).to(self.device)-mask
    
    def masked(self, A,mask):
        return (A/2+0.5)*mask*2-1
    
    def add_with_mask(self, A,B,mask):
        return ((A/2+0.5)*mask+(B/2+0.5)*(torch.ones(mask.shape).to(self.device)-mask))*2-1
    
    def addone_with_mask(self, A,mask):
        return ((A/2+0.5)*mask+(torch.ones(mask.shape).to(self.device)-mask))*2-1
    
    def partCombiner2(self, pillarl, pillarr, glass, balconie, sky, roof, railings, windows, mask, comb_op = 1):
        if comb_op == 0:
            # use max pooling, etc
            padvalue = -1
            hair = self.masked(sky, mask)
        else:
            # use min pooling,  etc
            padvalue = 1
            sky = self.addone_with_mask(sky, mask)
        IMAGE_SIZE = self.opt.fineSize
        ratio = IMAGE_SIZE / 256
        pillar_H = self.opt.pillar_H * ratio
        pillar_W = self.opt.pillar_W * ratio
        glass_H = self.opt.glass_H * ratio
        glass_W = self.opt.glass_W * ratio
        balconie_H = self.opt.balconie_H * ratio
        balconie_W = self.opt.balconie_W * ratio
        railings_H = self.opt.railings_H * ratio
        railings_W = self.opt.railings_W * ratio
        windows_H = self.opt.windows_H * ratio
        windows_W = self.opt.windows_W * ratio
        roof_H = self.opt.windows_H * ratio
        roof_W = self.opt.windows_W * ratio
        bs,nc,_,_ = pillarl.shape
        pillarl_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
        pillarr_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
        glass_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
        balconie_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
        for i in range(bs):
            center = self.center[i]#x,y
            pillarl_p[i] = torch.nn.ConstantPad2d((center[0,0] - pillar_W / 2, IMAGE_SIZE - (center[0,0]+pillar_W/2), center[0,1] - pillar_H / 2, IMAGE_SIZE - (center[0,1]+pillar_H/2)),padvalue)(pillarl[i])
            pillarr_p[i] = torch.nn.ConstantPad2d((center[1,0] - pillar_W / 2, IMAGE_SIZE - (center[1,0]+pillar_W/2), center[1,1] - pillar_H / 2, IMAGE_SIZE - (center[1,1]+pillar_H/2)),padvalue)(pillarr[i])
            glass_p[i] = torch.nn.ConstantPad2d((center[2,0] - glass_W / 2, IMAGE_SIZE - (center[2,0]+glass_W/2), center[2,1] - glass_H / 2, IMAGE_SIZE - (center[2,1]+glass_H/2)),padvalue)(glass[i])
            balconie_p[i] = torch.nn.ConstantPad2d((center[3,0] - balconie_W / 2, IMAGE_SIZE - (center[3,0]+balconie_W/2), center[3,1] - balconie_H / 2, IMAGE_SIZE - (center[3,1]+balconie_H/2)),padvalue)(balconie[i])
        if comb_op == 0:
            # use max pooling
            pillar = torch.max(pillarl_p, pillarr_p)
            pillar_glass = torch.max(pillar, glass_p)
            pillar_glass_balconie = torch.max(pillar_glass, balconie_p)
            result = torch.max(sky,pillar_glass_balconie)
        else:
            # use min pooling
            pillar = torch.min(pillarl_p, pillarr_p)
            pillar_glass = torch.min(pillar, glass_p)
            pillar_glass_balconie = torch.min(pillar_glass, balconie_p)
            result = torch.min(sky,pillar_glass_balconie)
        return result
    
    def partCombiner2_bg(self, pillarl, pillarr, glass, balconie, sky, bg, maskh, maskb, comb_op = 1):
        if comb_op == 0:
            # use max pooling, pad black for pillar etc
            padvalue = -1
            sky = self.masked(sky, maskh)
            bg = self.masked(bg, maskb)
        else:
            # use min pooling, pad white for pillar etc
            padvalue = 1
            sky = self.addone_with_mask(sky, maskh)
            bg = self.addone_with_mask(bg, maskb)
        IMAGE_SIZE = self.opt.fineSize
        ratio = IMAGE_SIZE / 256
        pillar_H = self.opt.pillar_H * ratio
        pillar_W = self.opt.pillar_W * ratio
        glass_H = self.opt.glass_H * ratio
        glass_W = self.opt.glass_W * ratio
        balconie_H = self.opt.balconie_H * ratio
        balconie_W = self.opt.balconie_W * ratio
        railings_H = self.opt.railings_H * ratio
        railings_W = self.opt.railings_W * ratio
        windows_H = self.opt.windows_H * ratio
        windows_W = self.opt.windows_W * ratio
        roof_H = self.opt.windows_H * ratio
        roof_W = self.opt.windows_W * ratio
        bs,nc,_,_ = pillarl.shape
        pillarl_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
        pillarr_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
        glass_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
        balconie_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
        railings_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
        windows_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
        roof_p = torch.ones((bs,nc,IMAGE_SIZE,IMAGE_SIZE)).to(self.device)
        for i in range(bs):
            center = self.center[i]#x,y
            pillarl_p[i] = torch.nn.ConstantPad2d((int(center[0,0] - pillar_W / 2), int(IMAGE_SIZE - (center[0,0]+pillar_W/2)), int(center[0,1] - pillar_H / 2), int(IMAGE_SIZE - (center[0,1]+pillar_H/2))),padvalue)(pillarl[i])
            pillarr_p[i] = torch.nn.ConstantPad2d((int(center[1,0] - pillar_W / 2), int(IMAGE_SIZE - (center[1,0]+pillar_W/2)), int(center[1,1] - pillar_H / 2), int(IMAGE_SIZE - (center[1,1]+pillar_H/2))), padvalue)(pillarr[i])
            glass_p[i] = torch.nn.ConstantPad2d((int(center[2,0] - glass_W / 2), int(IMAGE_SIZE - (center[2,0]+glass_W/2)), int(center[2,1] - glass_H / 2), int(IMAGE_SIZE - (center[2,1]+glass_H/2))),padvalue)(glass[i])
            balconie_p[i] = torch.nn.ConstantPad2d(
                (int(center[3,0] - balconie_W / 2), int(IMAGE_SIZE - (center[3,0]+balconie_W/2)), int(center[3,1] - balconie_H / 2), int(IMAGE_SIZE - (center[3,1]+balconie_H/2))),
                padvalue
            )(balconie[i])
        if comb_op == 0:
            pillars = torch.max(pillarl_p, pillarr_p)
            pillar_glass = torch.max(pillars, glass_p)
            pillar_glass_balconie = torch.max(pillar_glass, balconie_p)
            pillar_glass_balconie_sky = torch.max(sky,pillar_glass_balconie)
            result = torch.max(bg,pillar_glass_balconie_sky)
        else:
            pillars = torch.min(pillarl_p, pillarr_p)
            pillar_glass = torch.min(pillars, glass_p)
            pillar_glass_balconie = torch.min(pillar_glass, balconie_p)
            pillar_glass_balconie_sky = torch.min(sky,pillar_glass_balconie)
            result = torch.min(bg,pillar_glass_balconie_sky)
        return result
    
    def partCombiner3(self, building, sky, maskf, maskh, comb_op = 1):
        if comb_op == 0:
            # use max pooling, pad black etc
            padvalue = -1
            building = self.masked(building, maskf)
            sky = self.masked(sky, maskh)
        else:
            # use min pooling, pad white etc
            padvalue = 1
            building = self.addone_with_mask(building, maskf)
            sky = self.addone_with_mask(sky, maskh)
        if comb_op == 0:
            result = torch.max(building,sky)
        else:
            result = torch.min(building,sky)
        return result
    
    def getLocalParts(self,fakeAB):
        bs,nc,_,_ = fakeAB.shape #dtype torch.float32
        ncr = nc // self.opt.output_nc
        ratio = self.opt.fineSize // 256
        pillar_H = self.opt.pillar_H * ratio
        pillar_W = self.opt.pillar_W * ratio
        glass_H = self.opt.glass_H * ratio
        glass_W = self.opt.glass_W * ratio
        balconie_H = self.opt.balconie_H * ratio
        balconie_W = self.opt.balconie_W * ratio
        railings_H = self.opt.railings_H * ratio
        railings_W = self.opt.railings_W * ratio
        windows_H = self.opt.windows_H * ratio
        windows_W = self.opt.windows_W * ratio
        roof_H = self.opt.windows_H * ratio
        roof_W = self.opt.windows_W * ratio
        pillarl = torch.ones((bs,nc,pillar_H,pillar_W)).to(self.device)
        pillarr = torch.ones((bs,nc,pillar_H,pillar_W)).to(self.device)
        glass = torch.ones((bs,nc,glass_H,glass_W)).to(self.device)
        balconie = torch.ones((bs,nc,balconie_H,balconie_W)).to(self.device)
        railings = torch.ones((bs,nc,railings_H,railings_W)).to(self.device)
        windows = torch.ones((bs,nc,windows_H,windows_W)).to(self.device)
        roof = torch.ones((bs,nc,roof_H,roof_W)).to(self.device)
        for i in range(bs):
            center = self.center[i]
            pillarl[i] = fakeAB[i,:,center[0,1]-pillar_H//2:center[0,1]+pillar_H//2,center[0,0]-pillar_W//2:center[0,0]+pillar_W//2]
            pillarr[i] = fakeAB[i,:,center[1,1]-pillar_H//2:center[1,1]+pillar_H//2,center[1,0]-pillar_W//2:center[1,0]+pillar_W//2]
            glass[i] = fakeAB[i,:,center[2,1]-glass_H//2:center[2,1]+glass_H//2,center[2,0]-glass_W//2:center[2,0]+glass_W//2]
            balconie[i] = fakeAB[i,:,center[3,1]-balconie_H//2:center[3,1]+balconie_H//2,center[3,0]-balconie_W//2:center[3,0]+balconie_W//2]
            railings[i] = fakeAB[i,:,center[1,1]-railings_H//2:center[1,1]+railings_H//2,center[1,0]-railings_W//2:center[1,0]+railings_W//2]
            windows[i] = fakeAB[i,:,center[2,1]-windows_H//2:center[2,1]+windows_H//2,center[2,0]-windows_W//2:center[2,0]+windows_W//2]
            roof[i] = fakeAB[i,:,center[3,1]-roof_H//2:center[3,1]+roof_H//2,center[3,0]-roof_W//2:center[3,0]+roof_W//2]
        sky = (fakeAB/2+0.5)# * self.mask.repeat(1,ncr,1,1) * self.mask2.repeat(1,ncr,1,1) * 2 - 1
        bg = (fakeAB/2+0.5)# * (torch.ones(fakeAB.shape).to(self.device)-self.mask2.repeat(1,ncr,1,1)) * 2 - 1
        return pillarl, pillarr, glass, balconie, railings,windows, roof, sky, bg
    
    def getaddw(self,local_name):
        addw = 1
        if local_name in ['DLpillarl','DLpillarr','pillarl','pillarr']:
            addw = self.opt.addw_pillar
        elif local_name in ['DLglass','glass']:
            addw = self.opt.addw_glass
        elif local_name in ['DLbalconie','balconie']:
            addw = self.opt.addw_balconie
        elif local_name in ['DLrailings','railings']:   
            addw = self.opt.addw_railings
        elif local_name in ['DLwindows','windows']:
            addw = self.opt.addw_windows
        elif local_name in ['DLroof','roof']:       
            addw = self.opt.addw_roof               
        elif local_name in ['DLBG', 'bg']:
            addw = self.opt.addw_bg
        return addw
