import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from functools import partial

from srm_single_model import get_model
from srm_single_dataset import get_data

device = torch.device('cpu')
metrics = ['$', 'POP', 'LTR', 'EMP', 'MOR',  'AG']


def t2s(t):
    if isinstance(t, list) or isinstance(t, tuple):
        return [t2s(e) for e in t]
    else:
        return t.item()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def loss_batch(model, criterion, xb, yb, opt=None):
    xb = xb.to(device)
    for k,v in yb.items():
        yb[k] = v.to(device)

    wealth, density, literacy, employment, mortality, agriculture = model(xb)

    wealth_loss = criterion(wealth, yb['wealth'])
    wealth_acc  = accuracy(wealth, yb['wealth'], topk=(1,2))
    # wealth_pred = torch.argmax(wealth, dim=1)
    # wealth_acc  = (wealth_pred == yb['wealth']).sum()
    
    density_loss = criterion(density, yb['density'])
    density_acc  = accuracy(density, yb['density'], topk=(1,2))
    # pop_density_pred = torch.argmax(pop_density, dim=1)
    # pop_density_acc  = (pop_density_pred == yb['pop_density']).sum()

    literacy_loss = criterion(literacy, yb['literacy'])
    literacy_acc  = accuracy(literacy, yb['literacy'], topk=(1,2))
    # water_src_pred = torch.argmax(water_src, dim=1)
    # water_src_acc  = (water_src_pred == yb['water_src']).sum()

    employment_loss = criterion(employment, yb['employment'])
    employment_acc  = accuracy(employment, yb['employment'], topk=(1,2))
    # toilet_type_pred = torch.argmax(toilet_type, dim=1)
    # toilet_type_acc  = (toilet_type_pred == yb['toilet_type']).sum()

    #salary_loss = criterion(salary, yb['salary and wages'])
    #salary_acc  = accuracy(salary, yb['salary and wages'], topk=(1,2))
    # roof_pred = torch.argmax(roof, dim=1)
    # roof_acc  = (roof_pred == yb['roof']).sum()

    mortality_loss = criterion(mortality, yb['mortality'])
    mortality_acc  = accuracy(mortality, yb['mortality'], topk=(1,2))
    # cooking_fuel_pred = torch.argmax(cooking_fuel, dim=1)
    # cooking_fuel_acc  = (cooking_fuel_pred == yb['cooking_fuel']).sum()

    #drought_loss = criterion(drought, yb['drought'])
    #drought_acc  = accuracy(drought, yb['drought'], topk=(1,2))
    # drought_pred = torch.argmax(drought, dim=1)
    # drought_acc  = (drought_pred == yb['drought']).sum()

    

    #livestock_bin_loss = criterion(livestock_bin, yb['livestock_bin'])
    #livestock_bin_acc  = accuracy(livestock_bin, yb['livestock_bin'], topk=(1,2))
    # livestock_bin_pred = torch.argmax(livestock_bin, dim=1)
    # livestock_bin_acc  = (livestock_bin_pred == yb['livestock_bin']).sum()

    agriculture_loss = criterion(agriculture, yb['agriculture'])
    agriculture_acc  = accuracy(agriculture, yb['agriculture'], topk=(1,2))
    # agriculture_land_bin_pred = torch.argmax(agriculture_land_bin, dim=1)
    # agriculture_land_bin_acc  = (agriculture_land_bin_pred == yb['agriculture_land_bin']).sum()


    # print(wealth_acc.item()/xb.shape[0]*100, water_src_acc.item()/xb.shape[0]*100, toilet_type_acc.item()/xb.shape[0]*100, roof_acc.item()/xb.shape[0]*100)
    loss = 5 * wealth_loss + density_loss + literacy_loss + employment_loss +  mortality_loss + agriculture_loss
    # print(round(wealth_loss.item(), 2), round(water_src_loss.item(), 2), round(toilet_type_loss.item(), 2), round(roof_loss.item(), 2), round(cooking_fuel_loss.item(), 2), round(drought_loss.item(), 2), round(pop_density_loss.item(), 2), round(livestock_bin_loss.item(), 2), round(agriculture_land_bin_loss.item(), 2))

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    accs = t2s([wealth_acc, density_acc, literacy_acc, employment_acc, mortality_acc, agriculture_acc])

    return (loss.item(), accs, xb.shape[0])


def fit(epochs, model, opt, scheduler, criterion, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for idx, (xb, yb) in enumerate(train_dl):
            loss, accs, num = loss_batch(model, criterion, xb, yb, opt)
            if idx % 30 == 0:
                print(f'Epoch: {epoch}, T.Loss: {loss:.3f}, Accs: {[(metrics[i], e) for i,e in enumerate(accs)]}')

        model.eval()
        with torch.no_grad():
            losses, accs, nums = .0, [[.0,.0] for i in range(6)], .0
            for idx, (xb, yb) in enumerate(valid_dl):
                loss, acc, num = loss_batch(model, criterion, xb, yb)
                losses += loss
                nums += num
                for i in range(6):
                    for j in range(2):
                        accs[i][j] += acc[i][j]

        scheduler.step()
        print(f'Epoch: {epoch}, V.Loss: {(losses/len(valid_dl)):.3f}, Accs: {[(metrics[i], round(e[0]/len(valid_dl), 2), round(e[1]/len(valid_dl), 2)) for i,e in enumerate(accs)]}')


class Learner:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.wrapper.model = self.wrapper.model.to(device)

    def fit(self, epochs, lr, db, opt_fn, scheduler_fn):
        opt = opt_fn(self.wrapper.model.parameters(), lr=lr)
        scheduler = scheduler_fn(opt,
                                base_lr=lr/10,
                                max_lr=lr,
                                step_size_up=int(len(db.train_dl)*0.3),
                                step_size_down=int(len(db.train_dl)*0.7))

        criterion = F.cross_entropy
        fit(epochs, self.wrapper.model, opt, scheduler, criterion, db.train_dl, db.valid_dl)


def main():
    wrapper = get_model()
    opt_fn = optim.Adam
    cycle_fn = partial(optim.lr_scheduler.CyclicLR, mode='triangular2', cycle_momentum=False)
    # step_fn = partial(optim.lr_scheduler.StepLR, step_size=1, gamma=0.1)

    learn = Learner(wrapper)

    # # IMG_SZ = 224, BS: 32
    # db = get_data(img_sz=224, bs=64)

    # # Freeze: Features, 3 epochs
    # learn.wrapper.freeze_features()
    # print('\n\nPhase 1, 224, 64', learn.wrapper.grads)
    # learn.fit(2, 1e-2, db, opt_fn, cycle_fn)
    # learn.fit(2, 1e-3, db, opt_fn, cycle_fn)
    # torch.save(learn.wrapper.model.state_dict(), 'single_d121_ur_phase1_bal_4.pt')
    # # learn.wrapper.model.load_state_dict(torch.load('single_d121_ur_phase1_bal_4.pt'))

    # # Freeze: Partial(0.7), 3 epochs
    # learn.wrapper.partial_freeze_features(0.7)
    # print('\n\nPhase 2, 224, 64', learn.wrapper.grads)
    # learn.fit(2, 1e-4, db, opt_fn, cycle_fn)
    # torch.save(learn.wrapper.model.state_dict(), 'single_d121_ur_phase2_bal_4.pt')
    # # learn.wrapper.model.load_state_dict(torch.load('single_d121_ur_phase2.pt'))

    db = get_data(img_sz=224, bs=32)
    # # Freee: None, 3 Epochs
    learn.wrapper.freeze_features(False)
    # print('\n\nPhase 3, 224, 32', learn.wrapper.grads)
    learn.fit(2, 1e-4, db, opt_fn, cycle_fn)
    learn.fit(2, 1e-5, db, opt_fn, cycle_fn)
    torch.save(learn.wrapper.model.state_dict(), 'single_d121_ur_phase3_bal_4.pt')
    # # learn.wrapper.model.load_state_dict(torch.load('single_d121_ur_phase3.pt'))

    # IMG_SZ = 224, BS: 32
    # db = get_data(img_sz=224, bs=32)

    # print('\n\nPhase 4, 224, 32', learn.wrapper.grads)
    # learn.fit(1, 0.001, db, opt_fn, cycle_fn)
    # learn.fit(3, 0.1, db, opt_fn, cycle_fn)
    # torch.save(learn.wrapper.model.state_dict(), 'single_d121_phase4.pt')
    # learn.wrapper.model.load_state_dict(torch.load('single_d121_phase4.pt'))
    # learn.wrapper.freeze_features(False)

    # # IMG_SZ = 299, BS: 32
    # db = get_data(img_sz=299, bs=16)

    # print('\n\nPhase 4, 299, 16', learn.wrapper.grads)
    # learn.fit(2, 0.00001, db, opt_fn, cycle_fn)
    # torch.save(learn.wrapper.model.state_dict(), 'single_d121_ur_phase4.pt')

    db = get_data(img_sz=224, bs=64)
    learn.wrapper.model.load_state_dict(torch.load('single_d121_ur_phase3_bal_4.pt'))
    # Freee: None, 3 Epochs
    learn.wrapper.freeze_features(True)
    print('\n\nPhase 4, 224, 64', learn.wrapper.grads)
    learn.fit(2, 1e-5, db, opt_fn, cycle_fn)
    torch.save(learn.wrapper.model.state_dict(), 'single_d121_ur_phase3_bal_4_final.pt')
    # learn.wrapper.model.load_state_dict(torch.load('single_d121_ur_phase3.pt'))

if __name__ == "__main__":
    main()


