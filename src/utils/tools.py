import torch
import numpy as np

def get_batch_label(texts, prompt_text, label_map: dict):
    label_vectors = torch.zeros(0)

    for text in texts:
        label_vector = torch.zeros(len(prompt_text))
        
                                            
                                                                
                                          
        
        labels = text.split('-')                 
        for label in labels:
                                                        
            if label in label_map:
                class_name = label_map[label]
                if class_name in prompt_text:
                    label_vector[prompt_text.index(class_name)] = 1
            
                                              
            elif label in prompt_text:
                 label_vector[prompt_text.index(label)] = 1
            
                                        
            elif label == '0':
                if 'Normal' in prompt_text:
                    label_vector[prompt_text.index('Normal')] = 1
                elif 'normal' in prompt_text:        
                    label_vector[prompt_text.index('normal')] = 1

        label_vector = label_vector.unsqueeze(0)                     
        label_vectors = torch.cat([label_vectors, label_vector], dim=0)           

    return label_vectors

def get_prompt_text(label_map: dict):
    prompt_text = []
    for v in label_map.values():
                             
                                                            
        prompt_text.append(v) 
    return prompt_text



