import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import models

from srm_single_dataset import get_data

device = torch.device('cpu')


class Flatten(nn.Module):
    def forward(self, xb):
        return xb.view(xb.size(0), -1)


class SentinelResNet(nn.Module):
    def __init__(self, M, N, targets):
        super().__init__()
        self.rgb_features = nn.Sequential(*(list(M.children())[:-2] + [nn.AdaptiveAvgPool2d(1)]))
        self.nir_features = nn.Sequential(*(list(N.children())[:-2] + [nn.AdaptiveAvgPool2d(1)]))

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.wealth = nn.Linear(256, len(targets['wealth']['classes']))
        self.density = nn.Linear(256, len(targets['density']['classes']))
        self.literacy = nn.Linear(256, len(targets['literacy']['classes']))
        self.employment = nn.Linear(256, len(targets['employment']['classes']))
        #self.salary = nn.Linear(256, len(targets['salary']['classes']))
        self.mortality = nn.Linear(256, len(targets['mortality']['classes']))
        self.agriculture = nn.Linear(256, len(targets['agriculture']['classes']))
        
    def forward(self, xb):
        rgb_out = self.rgb_features(xb[0])
        nir_out = self.nir_features(xb[1])
        out = torch.cat([rgb_out, nir_out], dim=1)

        out = self.classifier(out)

        wealth = self.wealth(out)
        density = self.density(out)
        literacy = self.literacy(out)
        employment = self.employment(out)
        #salary = self.salary(out)
        mortality = self.mortality(out)
        agriculture = self.agriculture(out)

        return (wealth, density, literacy, employment,  mortality, agriculture)

class SentinelDenseNet(nn.Module):
    def __init__(self, M, targets):
        super().__init__()
        self.features = M.features

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Dropout(0.3),
        )
        self.wealth = nn.Linear(256, len(targets['wealth']['classes']))
        self.density = nn.Linear(256, len(targets['density']['classes']))
        self.literacy = nn.Linear(256, len(targets['literacy']['classes']))
        self.employment = nn.Linear(256, len(targets['employment']['classes']))
        #self.salary = nn.Linear(256, len(targets['salary']['classes']))
        self.mortality = nn.Linear(256, len(targets['mortality']['classes']))
        self.agriculture = nn.Linear(256, len(targets['agriculture']['classes']))
        
    def forward(self, xb):
        xb = self.features(xb)
        xb = self.classifier(xb)

        wealth = self.wealth(xb)
        density = self.density(xb)
        literacy = self.literacy(xb)
        employment = self.employment(xb)
        #salary = self.salary(xb)
        mortality = self.mortality(xb)
        agriculture = self.agriculture(xb)

        return (wealth, density, literacy, employment,  mortality, agriculture)
class Model:
    def __init__(self, M):
        self.model = M

    def __call__(self, xb):
        return self.model(xb)

    @staticmethod
    def freeze(L):
        for p in L.parameters(): p.requires_grad_(False)

    @staticmethod
    def unfreeze(L):
        for p in L.parameters(): p.requires_grad_(True)

    def freeze_features(self, arg=True):
        if arg:
            Model.freeze(self.model.features)
        else:
            Model.unfreeze(self.model.features)

    def freeze_classifier(self, arg=True):
        Model.freeze(self.model.classifier) if arg else Model.unfreeze(self.model.classifier)

    def partial_freeze_features(self, pct=0.2):
        sz = len(list(self.model.features.children()))
        point = int(sz * pct)

        for idx, child in enumerate(self.model.features.children()):
            Model.freeze(child) if idx <= point else Model.unfreeze(child)

    def summary(self):
        print('\n\n')
        for idx, (name, child) in enumerate(self.model.features.named_children()):
            print(f'{idx}: {name}-{child}')
            for param in child.parameters():
                print(f'{param.requires_grad}')

        for idx, (name, child) in enumerate(self.model.classifier.named_children()):
            print(f'{idx}: {name}-{child}')
            for param in child.parameters():
                print(f'{param.requires_grad}')
        print('\n\n')

    @property
    def grads(self):
        return ''.join(str(v.requires_grad)[0].upper() for k,v in self.model.named_parameters())


def get_model():
    d121 = models.densenet121(pretrained=True)

    model = SentinelDenseNet(d121, get_data().targets)
    wrapper = Model(model)

    return wrapper


