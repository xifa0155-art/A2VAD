from utils.layers import GraphConvolution, DistanceAdj
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from clip import clip

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

                                                               
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=0.0)                        
        self.ln_1 = LayerNorm(d_model)       
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)      
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None             
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None        
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x_in, padding_mask = x
        x = x_in + self.attention(self.ln_1(x_in), padding_mask)      
        x = x + self.mlp(self.ln_2(x))      
        return (x, padding_mask)

                                                                      
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class DualGraphEnhancement(nn.Module):
    def __init__(self, visual_width):
        super().__init__()
        width = int(visual_width / 2)
        
                                 
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()
        self.ln = nn.LayerNorm(visual_width)
        self.dropout = nn.Dropout(0.3)

                       
        nn.init.normal_(self.linear.weight, std=0.001)
        nn.init.zeros_(self.linear.bias)

    def adj4(self, x, seq_len):
        soft = nn.Softmax(dim=1)
                   
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True) + 1e-8
        x2 = torch.bmm(x, x.permute(0, 2, 1)) / torch.bmm(x_norm, x_norm.permute(0, 2, 1))
        
        output = torch.zeros_like(x2)
        
        for i in range(x.shape[0]):
            length = seq_len[i] if seq_len is not None else x.shape[1]
            tmp = x2[i, :length, :length]
            
                                                
            adj_mask = (tmp > 0.6).float() 
            adj2 = tmp * adj_mask + 1e-9
            output[i, :length, :length] = soft(adj2)
            
        return output

    def forward(self, x, lengths=None):
        residual = x 
        
        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1]).to(x.device)
        
                            
        x1 = self.gelu(self.gc1(x, adj))      
        x2 = self.gelu(self.gc3(x, disadj))      

        out = torch.cat((x1, x2), dim=2)
        out = self.linear(out)
        out = self.dropout(self.ln(out))            

        return residual + out

class CLIPVAD(nn.Module):
    def __init__(self, num_class, embed_dim, visual_length, visual_width, visual_head, visual_layers, 
                 attn_window, prompt_prefix, prompt_postfix, device, branch_mode='dual'):
        super().__init__()
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.device = device
        self.n_ctx = 3
        self.branch_mode = branch_mode

                                 
        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

                              
        self.linear = nn.Linear(visual_width, visual_width)

                            
        self.mlp_head = nn.Sequential(
            nn.Linear(visual_width, visual_width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(visual_width, 1)
        )

                          
        self.audio_adapter = nn.Sequential(
            nn.Conv1d(512, visual_width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(visual_width, visual_width, kernel_size=3, padding=1)
        )

                               
        self.visual_enhancement = DualGraphEnhancement(visual_width)

                             
        self.fusion_weight = nn.Sequential(       
            nn.Linear(visual_width * 2, visual_width),
            nn.Sigmoid()
        )
        self.fusion_res = nn.Sequential(           
            nn.Linear(visual_width * 2, visual_width),
            nn.LayerNorm(visual_width),                      
            QuickGELU(),
            nn.Linear(visual_width, visual_width)
        )

                             
        self.uncertainty_head = nn.Sequential(
            nn.Conv1d(visual_width, visual_width // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(visual_width // 2, 1, kernel_size=1)
        )

                                                                       
        self.mlp1 = nn.Sequential(nn.Linear(visual_width, visual_width), QuickGELU(), nn.Linear(visual_width, visual_width))
        self.mlp2 = nn.Sequential(nn.Linear(visual_width, visual_width), QuickGELU(), nn.Linear(visual_width, visual_width))

                 
        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for p in self.clipmodel.parameters():
            p.requires_grad = False
            
        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        
                             
        with torch.no_grad():
            prompt_prefix_txt = "a video of"
            tokenized = clip.tokenize(prompt_prefix_txt).to(device)
            embedding = self.clipmodel.token_embedding(tokenized).type(self.clipmodel.dtype)
            self.ctx = nn.Parameter(embedding[0, 1:1+self.n_ctx, :])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):           
                                              
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
                                      
                                                 
        weight_linear = self.fusion_weight[0]
        nn.init.zeros_(weight_linear.weight)
        nn.init.constant_(weight_linear.bias, -3.0)                      
        
                                                         
        for m in self.fusion_res.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001) 
                nn.init.zeros_(m.bias)
                
                                
        for m in self.audio_adapter.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.001) 
                nn.init.zeros_(m.bias)

                                                           
        for block in self.temporal.resblocks:
            if hasattr(block.attn, 'out_proj'):
                nn.init.zeros_(block.attn.out_proj.weight)
                nn.init.zeros_(block.attn.out_proj.bias)
            c_proj = block.mlp[-1]
            if isinstance(c_proj, nn.Linear):
                nn.init.zeros_(c_proj.weight)
                nn.init.zeros_(c_proj.bias)

    def build_attention_mask(self, attn_window):
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))            
        for i in range(int(self.visual_length / attn_window)):           
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0         
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0
        return mask

    def encode_video(self, images, lengths):
        images = images.to(torch.float)
        b, t, d = images.shape
        pos = torch.arange(t, device=self.device).unsqueeze(0).expand(b, -1)       
        pos_emb = self.frame_position_embeddings(pos.to(self.frame_position_embeddings.weight.device))
        x = images + pos_emb.to(images.device)
        x = x.permute(1, 0, 2)                                            
        x, _ = self.temporal((x, None))                                                                                
        x = x.permute(1, 0, 2)                           
        x = self.linear(x)
        return x

    def encode_text(self, text):
        prompts = [f"a video of {c}." for c in text]
        tokenized = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            embedding = self.clipmodel.token_embedding(tokenized).type(self.clipmodel.dtype)
        
           
        prefix = embedding[:, :1, :]   
        suffix = embedding[:, 1+self.n_ctx:, :]   
        ctx = self.ctx.unsqueeze(0).expand(len(text), -1, -1)
        prompt_emb = torch.cat([prefix, ctx, suffix], dim=1)
        
        x = prompt_emb + self.clipmodel.positional_embedding.type(self.clipmodel.dtype)
        x = x.permute(1, 0, 2)
        x = self.clipmodel.transformer(x)                           
        x = x.permute(1, 0, 2)
        x = self.clipmodel.ln_final(x).type(self.clipmodel.dtype)
        
        return x[torch.arange(x.shape[0]), tokenized.argmax(dim=-1)] @ self.clipmodel.text_projection

    def forward(self, visual, audio, text_labels, lengths, return_features=False):
                          
        visual_feat = self.encode_video(visual, lengths)            
        
                                       
        student_feat = self.visual_enhancement(visual_feat, lengths)
        
                                        
        sigma_in = student_feat.permute(0, 2, 1)
        log_sigma = self.uncertainty_head(sigma_in).squeeze(1)
        log_sigma = torch.clamp(log_sigma, min=-5.0, max=5.0)
        
                                         
                                                       
        dynamic_audio_gate = torch.sigmoid(log_sigma).unsqueeze(-1) 
        
                                                            
        is_teacher = (audio is not None and audio.sum() != 0)
        if is_teacher:
            aud_in = audio.permute(0, 2, 1)
            aud_feat = self.audio_adapter(aud_in).permute(0, 2, 1)
            cat_feat = torch.cat([student_feat, aud_feat], dim=-1)
            
                           
            w = self.fusion_weight(cat_feat)
            res = self.fusion_res(cat_feat)
            
                                                       
                                        
            combined_gate = w + dynamic_audio_gate - (w * dynamic_audio_gate)
            final_feat = student_feat + combined_gate * res
        else:
            final_feat = student_feat
            
                                 
        logits1 = self.mlp_head(final_feat + self.mlp2(final_feat))
        
             
                                                
        text_feat = self.encode_text(text_labels) 
        
                              
                    
        attn_score = torch.sigmoid(logits1) 
        global_feat = torch.bmm(attn_score.permute(0, 2, 1), final_feat)            
        global_feat = global_feat / (global_feat.norm(dim=-1, keepdim=True) + 1e-8)
        
                       
        text_feat_exp = text_feat.unsqueeze(0).expand(visual.shape[0], -1, -1)
        S_p = torch.bmm(text_feat_exp, global_feat.permute(0, 2, 1))
        S_p = F.softmax(S_p, dim=1)
        
                            
                                   
                                           
                                                             
        if self.branch_mode == 'dual_wo_csu':
            X_cp = text_feat_exp
        else:
            X_mp = S_p * global_feat
            X_combined = X_mp + text_feat_exp
            X_cp = self.mlp1(X_combined) + text_feat_exp
        
                       
        feat_norm = final_feat / (final_feat.norm(dim=-1, keepdim=True) + 1e-8)
        X_cp_norm = X_cp / (X_cp.norm(dim=-1, keepdim=True) + 1e-8)
        
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits2 = torch.bmm(feat_norm, X_cp_norm.permute(0, 2, 1)) * logit_scale
                                         
        
        if return_features:
            feat_dict = {
                                             
                                        
                "X_v": visual.detach(),
                "X_a": audio.detach() if audio is not None else None,

                                                                                            
                "X_vis": student_feat.detach(),

                                                                             
                "X_fused": final_feat.detach(),
            }
            return text_feat, logits1, logits2, feat_dict

        return text_feat, logits1, logits2