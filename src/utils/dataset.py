import torch.utils.data as data
import numpy as np
import os
import torch

class XDDataset(data.Dataset):
    def __init__(self, visual_length, list_file, is_test, label_map):
        self.visual_length = visual_length
        self.label_map = label_map
                                   
        self.visual_base = './Dataset/features/XD_CLIP/XD-Violence'
        self.audio_base = './Dataset/features/XD_Wav2CLIP'
        
        with open(list_file, 'r') as f:
            self.list = f.readlines()
        
                               
        if 'file' in self.list[0] or 'path' in self.list[0]:
            self.list = self.list[1:]

    def __getitem__(self, index):
        line = self.list[index].strip()                  
        parts = line.split(',')
        full_path = parts[0]
        file_name = os.path.basename(full_path)
        
                                            
        if not file_name.endswith('.npy'):
            file_name = file_name + '.npy'
            
        label_str = parts[1] if len(parts) > 1 else '0'
        
                              
                           
                     
        candidate_dirs = ['XDTrainClipFeatures', 'XDTestClipFeatures']
        
        vis_path = None
        found = False
        
        for sub_dir in candidate_dirs:
            p = os.path.join(self.visual_base, sub_dir, file_name)
            if os.path.exists(p):
                vis_path = p
                                   
                aud_path = os.path.join(self.audio_base, sub_dir, file_name)
                found = True
                break
        
                                             
                                       
        if not found:
            raise FileNotFoundError(
                f"\n[Critical Error] 找不到视觉特征文件: {file_name} \n"
                f"请检查路径: {self.visual_base} 下的子目录。"
            )

                                                         
        vis = np.load(vis_path)
        
                                         
        if os.path.exists(aud_path):
            aud = np.load(aud_path)
        else:
                                       
            aud = np.zeros((vis.shape[0], 512))
                                                                                                 

        T = vis.shape[0]

                                     
        if T > self.visual_length:
            sample_idx = np.linspace(0, T - 1, self.visual_length).astype(int)
            vis = vis[sample_idx]
            aud = aud[sample_idx]
            actual_len = self.visual_length
        else:
            sample_idx = np.arange(T).astype(int)
            pad_len = self.visual_length - T

            vis = np.pad(vis, ((0, pad_len), (0, 0)), 'constant')
            aud = np.pad(aud, ((0, pad_len), (0, 0)), 'constant')

            pad_idx = np.full(pad_len, -1, dtype=int)
            sample_idx = np.concatenate([sample_idx, pad_idx], axis=0)

            actual_len = T

        return (
            torch.from_numpy(vis).float(),
            torch.from_numpy(aud).float(),
            label_str,
            actual_len,
            T,
            torch.from_numpy(sample_idx).long()
        )

    def __len__(self):
        return len(self.list)