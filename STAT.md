
# Model Performance

Here are the generative performances under different training settings and model configs.

Generative performance of unconditional models.

|  Model Name | Type | Params | FID | IS  | Training Data | BS  | Iters |
| :---------  | :--  | :----: | :-: | :-: | :----------:  | :-: | :---: |
| [GET-T](https://drive.google.com/file/d/1rDw5A34ZnTajQZLSb_7fGkUfQU6viwq8/view?usp=sharing)  | Uncond | 8.6M | 15.23 | 8.40 | 1M | 128 | 800k |
| [GET-M](https://drive.google.com/file/d/1bAcRl0dWDxzIkm3sZBzBMABw6y78mVcQ/view?usp=sharing)  | Uncond | 19.2M | 10.81 | 8.77 | 1M | 128 | 800k |
| [GET-S](https://drive.google.com/file/d/1rN2rD7WUDaJaL3uRU5eX14wKQAg7Zy8z/view?usp=sharing)  | Uncond | 37.2M | 7.99 | 9.05 | 1M | 128 | 800k |
| [GET-B](https://drive.google.com/file/d/1k7qMLfqxctFNldsUSuapLP96oIllwZ1H/view?usp=sharing)  | Uncond | 62.2M | 7.39 | 9.17 | 1M | 128 | 800k |
| [GET-B+](https://drive.google.com/file/d/1jUE1lqs0qsbqbLyl9nmROcxrETDUXx25/view?usp=sharing)  | Uncond | 83.5M | 7.21 | 9.07 | 1M | 128 | 800k |

Generative performance of class-conditional models.

|  Model Name | Type | Params | FID | IS  | Training Data | BS  | Iters |
| :---------  | :--  | :----: | :-: | :-: | :----------:  | :-: | :---: |
| [GET-B](https://drive.google.com/file/d/1BPPtWpoXVexgozaKAiZRx0N5egweHrFH/view?usp=sharing)  | Cond | 62.2M | 6.23 | 9.42 | 1M | 256 | 800k |
| [GET-B](https://drive.google.com/file/d/1DH8cN70OucFRoWsXJvK4vgyIrctAcoFN/view?usp=sharing)  | Cond | 62.2M | 5.66 | 9.63 | 2M | 256 | 1.2M |

There is a clear scaling for Generative Equilibrium Transformers. Mostly, FID has a **log linear** relation w.r.t. the input money (=Training FLOPs/Time/Data/Params) when fixing other dimensions. Scaling up the training data, better supervision from the perceptual loss/teacher model, more training FLOPs/larger batch size/longer training schedule can lead to better results, as demonstrated.

Ideally, when scaling up training data, the model size and training flops need to adjust accordingly to achieve the best training efficiency. Nonetheless, restricted by our computing resources, we cannot derive the *exact* scaling law for compute-optimal models as shown in [Chinchilla](https://arxiv.org/abs/2203.15556). Despite the compute restriction, our observation still shows that GET's scaling law suggests **much more compact compute-optimal models** than ViTs, which is ideal for memory-bounded deployment, like today's LLM.

In particular, this work shows that **memorizing sufficient "regular" data pairs can lead to a good generative model**, no matter for GET or ViT. The differences can be data efficiency and training efficiency (we assume there are much better training strategies though). 

Here, the term "regular" means the latent-image pairs are easy to learn, e.g., sampled from a pretrained model. Please note that the randomly paired latent-code, for example, shuffling the latent-image pairs in the training set, cannot be memorized by a model at a constant learning rate (we use a fixed learning rate to train models over massive pairs in the above experiments), as the loss curve is non-decreasing. This implies that **the learnability of pairing can be a strong measurement of pairing quality**.

