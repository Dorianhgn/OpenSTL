import torch
import sys

def test_mamba_installation():
    print("--- 🚀 DÉBUT DU DIAGNOSTIC MAMBA2 (DIMENSIONS CORRIGÉES) ---")

    if not torch.cuda.is_available():
        print("❌ Pas de GPU.")
        sys.exit(1)

    device = "cuda"
    # On force bfloat16 si possible, sinon float16 (Mamba préfère BF16 sur Ampere/Hopper)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"   - GPU: {torch.cuda.get_device_name(0)}")
    print(f"   - Dtype: {dtype}")

    try:
        from mamba_ssm import Mamba2
        print(f"   ✅ Mamba2 importé.")
    except ImportError as e:
        print(f"❌ Erreur import: {e}")
        sys.exit(1)

    print(f"\n[TEST DE CALCUL]")
    try:
        # --- CONFIGURATION QUI MARCHE ---
        # Règle : (d_model * expand) / headdim doit être un multiple de 8 (souvent)
        # Ici : (256 * 2) / 64 = 8 têtes. C'est parfait.
        
        batch = 2
        length = 64
        dim = 256       # Augmenté à 256 pour avoir 8 têtes
        headdim = 64
        expand = 2
        
        print(f"   - Config: d_model={dim}, expand={expand}, headdim={headdim}")
        print(f"   - Nombre de têtes calculé: {(dim*expand)//headdim}")

        model = Mamba2(
            d_model=dim,
            d_state=64,
            d_conv=4,
            expand=expand,
            headdim=headdim
        ).to(device=device, dtype=dtype)
        
        # .contiguous() est important pour éviter les erreurs de stride
        x = torch.randn(batch, length, dim, device=device, dtype=dtype).contiguous()
        
        y = model(x)
        
        assert y.shape == x.shape
        print(f"   ✅ Forward pass réussi ! Output shape: {y.shape}")
        print("\n--- 🏁 INSTALLATION VALIDÉE ---")
        
    except Exception as e:
        print(f"   ❌ ERREUR : {e}")
        # Si ça plante encore ici, c'est que l'installation est vraiment cassée.
        sys.exit(1)

if __name__ == "__main__":
    test_mamba_installation()