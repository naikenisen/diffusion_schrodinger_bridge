# Pont de Schrödinger Diffusif avec Applications à la Modélisation Générative basée sur le Score

Ce dépôt contient l'implémentation de l'article "Pont de Schrödinger Diffusif avec Applications à la Modélisation Générative basée sur le Score".

## Qu'est-ce qu'un pont de Schrödinger ?

Le problème du Pont de Schrödinger (SB) est un problème classique en mathématiques appliquées, contrôle optimal et probabilité ; voir [1, 2, 3]. En temps discret, il prend la forme dynamique suivante : on considère une densité de référence p(x<sub>0:N</sub>) décrivant le processus d'ajout de bruit aux données. On cherche à trouver p\*(x<sub>0:N</sub>) telle que p\*(x<sub>0</sub>) = p<sub>data</sub>(x<sub>0</sub>) et p\*(x<sub>N</sub>) = p<sub>prior</sub>(x<sub>N</sub>), tout en minimisant la divergence de Kullback-Leibler entre p\* et p. Dans ce travail, nous introduisons le **Pont de Schrödinger Diffusif** (DSB), un nouvel algorithme utilisant des approches de score-matching [4] pour approximer l'algorithme *Iterative Proportional Fitting*, une méthode itérative pour résoudre le problème SB. DSB peut être vu comme un raffinement des méthodes existantes de modélisation générative basée sur le score.

## Applications

Une application prometteuse de cette approche est le virtual staining en histopathologie, par exemple la conversion d'images colorées HES (Hématoxyline-Éosine-Safran) vers des images IHC. Le DSB permet d'apprendre une transformation probabiliste entre deux distributions d'images, ici entre la distribution des coupes HES et celle des coupes IHC, tout en garantissant une correspondance réaliste et contrôlée. Cela ouvre la voie à la génération d'images IHC virtuelles à partir de coupes HES, facilitant l'analyse biomédicale sans recourir à des colorations coûteuses ou destructives, et en conservant la structure et l'information du tissu original.

```
    @article{de2021diffusion,
              title={Diffusion Schr$\backslash$" odinger Bridge with Applications to Score-Based Generative Modeling},
              author={De Bortoli, Valentin et Thornton, James et Heng, Jeremy et Doucet, Arnaud},
              journal={arXiv preprint arXiv:2106.01357},
              year={2021}
            }
```
