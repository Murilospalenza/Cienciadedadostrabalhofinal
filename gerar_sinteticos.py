"""Gera dados sintéticos com CTAB-GAN+ ou Tab-PE.

Uso (cada modelo roda no seu próprio venv):
  python gerar_sinteticos.py --model ctabgan --dataset adult --epochs 300
  python gerar_sinteticos.py --model tabpe   --dataset adult --epsilon 1.0
"""
import os
os.environ["MPLBACKEND"] = "Agg"   # o Jupyter exporta um backend que não existe fora dele

import argparse
import glob
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["ctabgan", "tabpe"], required=True)
p.add_argument("--dataset", required=True)
p.add_argument("--base", default=str(Path.home() / "cmla_trabalho"))
p.add_argument("--epochs", type=int, default=150)      # CTAB-GAN+ (150 = padrão da biblioteca, como no artigo)
p.add_argument("--epsilon", type=float, default=1.0)   # Tab-PE
p.add_argument("--iters", type=int, default=30)        # Tab-PE (precisa ser > 5)
p.add_argument("--n-syn", type=int, default=0)         # Tab-PE; 0 = mesmo tamanho do real
p.add_argument("--label", default="")                  # Tab-PE; padrão = última coluna
p.add_argument("--seed", type=int, default=42)
p.add_argument("--force", action="store_true")
args = p.parse_args()

BASE = Path(args.base)
SAIDA = BASE / "sinteticos"
SAIDA.mkdir(parents=True, exist_ok=True)
sufixo = (f"ctabgan_ep{args.epochs}" if args.model == "ctabgan"
          else f"tabpe_eps{args.epsilon:g}" + (f"_n{args.n_syn}" if args.n_syn else ""))
destino = SAIDA / f"{args.dataset}_{sufixo}.csv"
if destino.exists() and not args.force:
    print(f"{destino.name} já existe, pulando")
    sys.exit(0)

np.random.seed(args.seed)

# ---------- dados e tipos de coluna ----------
df = pd.read_csv(BASE / "dados" / f"{args.dataset}.csv")
for c in df.columns:                        # texto que na verdade é número (ex.: TotalCharges do Telco)
    if not pd.api.types.is_numeric_dtype(df[c]):
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() > 0.9:
            df[c] = num
df = df.dropna().reset_index(drop=True)
df.to_csv(BASE / "dados" / f"{args.dataset}_limpo.csv", index=False)   # usar este como --real
label = args.label or df.columns[-1]

cat, inteiras, floats = [], [], []
for c in df.columns:
    s = df[c]
    if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
        cat.append(c)                       # texto, category, bool
    else:
        e_inteira = (s % 1 == 0).all()
        if e_inteira and s.nunique() <= 10:
            cat.append(c)                   # inteiro com poucos valores = categoria
        elif e_inteira:
            inteiras.append(c)
        else:
            floats.append(c)
print(f"{args.dataset}: {df.shape} | cat={len(cat)} int={len(inteiras)} float={len(floats)} | label={label}")

tmp = SAIDA / f"_tmp_{args.dataset}_{sufixo}"
tmp.mkdir(exist_ok=True)
csv_real = tmp / "real.csv"
df.to_csv(csv_real, index=False)
t0 = time.time()

# ---------- CTAB-GAN+ ----------
if args.model == "ctabgan":
    import torch
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(BASE / "modelos" / "CTAB-GAN-Plus"))
    from model.ctabgan import CTABGAN

    mixed = {}
    for c in inteiras + floats:
        frac = df[c].value_counts(normalize=True)
        if frac.iloc[0] >= 0.4 and df[c].nunique() > 10:   # valor especial dominante (ex.: 0)
            mixed[c] = [float(frac.index[0])]

    print("device:", "cuda" if torch.cuda.is_available() else "CPU (vai ser lento)")
    modelo = CTABGAN(
        raw_csv_path=str(csv_real),
        categorical_columns=cat,
        log_columns=[],
        mixed_columns=mixed,
        general_columns=[],
        non_categorical_columns=[],
        integer_columns=inteiras,
        problem_type={None: None},   # sem split: treina com o dataset inteiro
    )
    modelo.synthesizer.epochs = args.epochs
    modelo.fit()                      # a barra por epoch é do próprio CTAB-GAN+
    syn = modelo.generate_samples()

# ---------- Tab-PE ----------
else:
    from pe.data import TabularCSV
    from pe.runner import PE
    from pe.population import PEPopulation, CompositePopulation
    from pe.api import TabularAPI
    from pe.embedding import TabularEmbedding
    from pe.histogram import NearestNeighbors
    from pe.callback import SaveTabToCSV
    from pe.callback.callback import Callback
    from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME
    from pe.logger import CSVPrint

    class SalvarFinal(SaveTabToCSV):
        """Salva só as amostras finais (descarta as variações intermediárias)."""
        def __call__(self, syn_data):
            return super().__call__(syn_data.filter({VARIATION_API_FOLD_ID_COLUMN_NAME: -1}))

    class Barra(Callback):
        def __init__(self, total):
            self.b = tqdm(total=total, desc="Tab-PE iterações")
        def __call__(self, syn_data):
            self.b.update(1)

    meta = {
        "cat_columns": [c for c in cat if c != label],
        "int_columns": [c for c in inteiras if c != label],
        "float_columns": [c for c in floats if c != label],
        "label_columns": [label],
    }
    json.dump(meta, open(tmp / "meta.json", "w"))
    priv = TabularCSV(csv_path=str(csv_real), metadata_path=str(tmp / "meta.json"))
    info = priv.get_tab_info()

    n_syn = args.n_syn or len(df)
    api = TabularAPI(info=info, mutation_rate_init=0.5, mutation_rate_final=0.01,
                     decay_type="polynomial", gamma=0.2, num_iterations=args.iters)
    hist = NearestNeighbors(embedding=TabularEmbedding(info=info), mode="L2",
                            lookahead_degree=0, backend="torch")
    pop1 = PEPopulation(api=api, initial_variation_api_fold=0, next_variation_api_fold=1,
                        keep_selected=False, selection_mode="sample", histogram_threshold=0)
    pop2 = PEPopulation(api=api, initial_variation_api_fold=3, next_variation_api_fold=3,
                        keep_selected=True, selection_mode="rank")
    pop = CompositePopulation(populations=[pop1] * 5 + [pop2] * (args.iters - 5))

    n = len(priv.data_frame)
    ckpt = tmp / "ckpt"
    shutil.rmtree(ckpt, ignore_errors=True)
    PE(priv_data=priv, population=pop, histogram=hist,
       callbacks=[SalvarFinal(output_folder=str(tmp / "syn")), Barra(args.iters)],
       loggers=[CSVPrint(output_folder=str(tmp))],
    ).run(num_samples_schedule=[n_syn] * args.iters,
          delta=1.0 / n / np.log(n), epsilon=args.epsilon, checkpoint_path=str(ckpt))
    syn = pd.read_csv(sorted(glob.glob(str(tmp / "syn" / "*.csv")))[-1])

# ---------- salvar ----------
syn = syn[list(df.columns)]
for c in df.columns:                        # volta ao tipo inteiro do dado real (ex.: 2.0 -> 2)
    if pd.api.types.is_integer_dtype(df[c]):
        col = pd.to_numeric(syn[c], errors="coerce").fillna(df[c].median())
        syn[c] = col.round().astype(df[c].dtype)
syn.to_csv(destino, index=False)
shutil.rmtree(tmp, ignore_errors=True)
print(f"OK: {destino} {syn.shape} em {(time.time() - t0) / 60:.1f} min")
