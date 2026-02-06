# 🐍 Mamba2 & Causal-Conv1d : Spécifications Techniques

Ce document détaille les contraintes architecturales strictes imposées par l'implémentation CUDA optimisée de Mamba2 (`mamba-ssm` + `causal-conv1d`).

> **⚠️ À L'ATTENTION DE GITHUB COPILOT / DEVELOPERS**
> Lors de la définition des hyper-paramètres du modèle Mamba2, vous **devez** respecter les règles d'alignement mémoire ci-dessous pour éviter les crashs CUDA (`RuntimeError`).

## 1. La "Règle des 8" (The Rule of 8)

Le kernel CUDA `causal-conv1d` exige un alignement mémoire spécifique pour utiliser les Tensor Cores. 

**La contrainte :** Le nombre de têtes (`n_heads`) doit être un **multiple de 8**.

### Formule de vérification
```python
n_heads = (d_model * expand) / headdim
assert n_heads % 8 == 0, "n_heads must be a multiple of 8 for causal-conv1d kernel!"
```

### ✅ Configurations Valides (Safe)

* `d_model=256`, `expand=2`, `headdim=64` → 8 têtes (OK)
* `d_model=512`, `expand=2`, `headdim=64` → 16 têtes (OK)
* `d_model=128`, `expand=2`, `headdim=32` → 8 têtes (OK)

### ❌ Configurations Invalides (Crash CUDA)

* `d_model=128`, `expand=2`, `headdim=64` → 4 têtes (**CRASH**)
* `d_model=192`, `expand=2`, `headdim=64` → 6 têtes (**CRASH**)

---

## 2. Symptômes d'une mauvaise configuration

Si cette règle n'est pas respectée, l'entraînement plantera immédiatement lors du premier forward pass avec cette erreur cryptique :

```text
RuntimeError: causal_conv1d with channel last layout requires strides (x.stride(0) and x.stride(2)) to be multiples of 8
```

ou

```text
Exception raised from causal_conv1d_fwd ... requires strides ...
```

---

## 3. Solutions de contournement (peu recommandé)

Si votre architecture impose une dimension qui viole la "Règle des 8" (ex: vous êtes obligés d'avoir `d_model=192`), vous avez deux options :

1. **Ajuster `headdim` :** Réduisez la dimension de la tête (ex: passer de 64 à 48 ou 32) pour que le ratio retombe sur un multiple de 8.
2. **Désactiver le kernel optimisé (Mode Fallback) :**
Si la performance n'est pas critique, désinstallez le package `causal-conv1d`. Mamba basculera automatiquement sur une implémentation PyTorch native (`torch.nn.Conv1d`) qui accepte n'importe quelle dimension.
```bash
pip uninstall causal-conv1d
```

---

## 4. Initialisation des Tenseurs

Pour les tests unitaires ou l'inférence manuelle, assurez-vous toujours que les tenseurs d'entrée sont **contigus** en mémoire :

```python
# Toujours ajouter .contiguous() avant de passer dans Mamba2
x = torch.randn(batch, seq, dim, device="cuda", dtype=dtype).contiguous()
y = model(x)
```
