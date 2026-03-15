# smiles_augment.py — SMILES Randomization for Data Augmentation
# ================================================================
# RDKit의 MolToSmiles(doRandom=True)를 사용하여
# 동일 분자의 다양한 SMILES 표현을 생성.
# 5,700 → 실질 50,000+ 효과. Scaffold Split 방어력 향상.
# ================================================================

from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles


def randomize_smiles(smiles, n_augment=10, include_canonical=True):
    """SMILES 1개 → n_augment개의 랜덤 SMILES 변형 생성
    
    Args:
        smiles: 원본 SMILES 문자열
        n_augment: 생성할 변형 수
        include_canonical: True면 canonical 형태를 첫 번째로 포함
    
    Returns:
        list[str]: 고유한 SMILES 변형 리스트 (최대 n_augment개)
    """
    mol = MolFromSmiles(smiles)
    if mol is None:
        return [smiles]  # 파싱 실패 시 원본 반환
    
    results = set()
    
    # canonical SMILES 추가
    canonical = MolToSmiles(mol)
    if include_canonical:
        results.add(canonical)
    
    # 랜덤 SMILES 생성 (최대 n_augment * 3번 시도)
    max_attempts = n_augment * 3
    for _ in range(max_attempts):
        if len(results) >= n_augment:
            break
        rand_smi = MolToSmiles(mol, doRandom=True)
        if rand_smi:
            results.add(rand_smi)
    
    return list(results)[:n_augment]


def augment_smiles_list(smiles_list, n_augment=10):
    """SMILES 리스트 전체를 증강
    
    Args:
        smiles_list: 원본 SMILES 리스트
        n_augment: 각 SMILES당 생성할 변형 수
    
    Returns:
        list[tuple[str, str]]: (augmented_smiles, original_smiles) 쌍 리스트
    """
    augmented = []
    for smi in smiles_list:
        variants = randomize_smiles(smi, n_augment=n_augment)
        for v in variants:
            augmented.append((v, smi))  # (변형, 원본)
    return augmented


def augment_molecules(molecules, n_augment=10):
    """분자 딕셔너리 리스트를 증강 (DB 형식 유지)
    
    Args:
        molecules: [{'smiles': ..., 'odor_labels': [...], ...}, ...]
        n_augment: 각 분자당 생성할 변형 수
    
    Returns:
        list[dict]: 증강된 분자 리스트 (원본 포함)
    """
    augmented = []
    for mol in molecules:
        smiles = mol.get('smiles', '')
        if not smiles:
            augmented.append(mol)
            continue
        
        variants = randomize_smiles(smiles, n_augment=n_augment)
        for v in variants:
            aug_mol = dict(mol)  # shallow copy
            aug_mol['smiles'] = v
            aug_mol['_original_smiles'] = smiles  # 원본 추적용
            augmented.append(aug_mol)
    
    return augmented


if __name__ == '__main__':
    # 테스트
    test_smiles = [
        ('O=Cc1ccc(O)c(OC)c1', 'Vanillin'),
        ('CC(=O)OCC=C(C)C', 'Linalyl acetate'),
        ('CC(C)=CCCC(C)=CC=O', 'Citral'),
        ('CCO', 'Ethanol'),
    ]
    
    print("=" * 60)
    print("  SMILES Randomization Test")
    print("=" * 60)
    
    for smi, name in test_smiles:
        variants = randomize_smiles(smi, n_augment=10)
        unique = len(set(variants))
        print(f"\n  {name} ({smi})")
        print(f"  Generated {len(variants)} variants ({unique} unique):")
        for i, v in enumerate(variants[:5]):
            marker = " (canonical)" if v == MolToSmiles(MolFromSmiles(smi)) else ""
            print(f"    {i+1}. {v}{marker}")
        if len(variants) > 5:
            print(f"    ... and {len(variants)-5} more")
    
    # 증강 효과 테스트
    print(f"\n\n  Augmentation Stats:")
    for n in [5, 10, 20]:
        total = 0
        for smi, _ in test_smiles:
            total += len(randomize_smiles(smi, n_augment=n))
        print(f"    n_augment={n:2d}: {len(test_smiles)} → {total} samples ({total/len(test_smiles):.1f}x)")
