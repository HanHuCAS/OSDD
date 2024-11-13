import os
import glob
import json
import torch
from model_baseline import convnext_base,convnext_large
from dataloader import get_data_loader
import numpy as np
import torch.nn.functional as F

def get_features(dataloader, model):
    imgs_name = []
    with torch.no_grad():
        vecs = torch.zeros(1536, len(dataloader))
        for i, (img, img_name) in enumerate(dataloader):
            img = img.cuda()
            feats = model.forward_features(img)
            feats = F.normalize(feats)
            vecs[:, i] = feats.cpu().squeeze()
            imgs_name.extend(img_name)
    return vecs.numpy(), imgs_name


def main():
    # get images in probe set
    data_root = '/data'
    probe_imgs = glob.glob(os.path.join(data_root, 'probe/*'))
    gallery_imgs = glob.glob(os.path.join(data_root, 'gallery/*'))
    print('probe:{},gallery:{}'.format(len(probe_imgs), len(gallery_imgs)))

    probe_loader = get_data_loader(data_root=data_root,
                                   batch_size=1,
                                   num_workers=8,
                                   data_source='probe')

    gallery_loader = get_data_loader(data_root=data_root,
                                     batch_size=1,
                                     num_workers=8,
                                     data_source='gallery')

    # build your own model
    model=convnext_large()
    pretrained_dict = torch.load('./scripts/weights.pth', map_location=lambda storage, loc: storage.cuda())
    model.load_state_dict(pretrained_dict['model'], strict=True)
    del pretrained_dict
    model = model.cuda()
    model.eval()

    # inference
    probe_feats, probe_imgs_name = get_features(probe_loader, model)
    gallery_feats, gallery_imgs_name = get_features(gallery_loader, model)

    print('probe:', probe_feats.shape)
    print('gallery:', gallery_feats.shape)

    sim = np.dot(gallery_feats.T, probe_feats)
    print('sim:', sim.shape)

    # search, rank
    ranks = np.argsort(-sim, axis=0)

    # find distance between probe and gallery
    gp=[]
    for pi in range(probe_feats.shape[1]):
        pilen=[]
        for i in range(gallery_feats.shape[1]):
            pilen.append(np.sum(abs(gallery_feats[:,i]-probe_feats[:,pi])))
        gp.append([pi,np.array(pilen).mean()])

    # find new_classes
    m=np.array(gp)[:,1].mean()
    pred=np.zeros((len(probe_imgs_name)))
    pred[np.where(np.array(gp)[:,1]>m)[0]]=1
    # print(sum(pred==tp),len(pred))

    # write to output json
    results = {}
    k = 5
    for i in range(sim.shape[1]):
        topk = {}
        results[probe_imgs_name[i]] = {}
        for j in range(k):
            topk[gallery_imgs_name[ranks[j, i]]] = '{:.4f}'.format(sim[ranks[j, i], i])
        results[probe_imgs_name[i]]['similarity'] = topk
        results[probe_imgs_name[i]]['novelty'] = pred[i]

    # write result into result.json
    with open('/profile/result.json', 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()