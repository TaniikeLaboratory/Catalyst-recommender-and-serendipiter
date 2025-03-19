import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

# Catalysts number for file name
cat_num = 376

# threshold for deciding performance class
thr = 13

# weight of degree of belief for one time observation
weight1 = 0.2
weight2 = 0.2
weight3 = 0.2

# [disimprove, unchange, improve]
H_value = [-1, 0, 1]
# [positive, non-positive]
C = [1, 0]


def compare(a, b):
    a1 = copy.deepcopy(a)
    b1 = copy.deepcopy(b)
    c = []
    for x in a:
        if x in b1:
            b1.remove(x)
            c.append(x)
    for x in b:
        if x in a1:
            a1.remove(x)
    c = list(set(c))
    return [a1, b1, c]


def Dempster(m1, m2):
    scale = 1.0
    new_mass = [0.0] * len(m1)
    for s0 in range(len(m1) - 1):
        buffer = []
        for s1 in range(len(m1) - 1):
            if s0 != s1:
                buffer.append(m1[s0] * m2[s1])
        scale -= sum(buffer)
        new_mass[s0] = m1[s0] * m2[s0] + m1[s0] * m2[-1] + m1[-1] * m2[s0]
    new_mass[-1] = m1[-1] * m2[-1]
    new_mass = [x / scale for x in new_mass]
    return new_mass


def evidence_apply(mass_x, Cat_mass_x):
    m1 = Cat_mass_x[j]
    m2_origin = mass_x[idx_former][idx_latter]
    if m2_origin[-1] == 1.0:
        return
    m2 = [0.0] * len(C)
    m2.append(1.0)
    for m in range(len(H_value)):
        if (C2y_class[i] == 1 and H_value[m] in [1, 0]):
            predicted_C2y = 1
        elif (C2y_class[i] == 0 and H_value[m] in [-1, 0]):
            predicted_C2y = 0
        else:
            continue
        idx_C2y = C.index(predicted_C2y)
        m2[idx_C2y] += m2_origin[m]
    m2[-1] = 1.0 - sum(m2[:-1])
    updated_mass = Dempster(m1, m2)
    m1[:] = updated_mass
    return


# Original library
M = ["Li", "Na", "Mg", "K", "Ca", "Ti", "V", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Sr",
     "Y", "Zr", "Mo", "Pd", "Cs", "Ba", "La", "Ce", "Nd", "Eu", "Tb", "Hf", "W", "none"]
S = ["MgO", "Al2O3", "SiO2", "CaO", "TiO2", "ZrO2", "BaO", "La2O3", "CeO2"]
sort_key = (M + S).index

# Library of pairs
unary = M + S
binary = []
for i in range(len(M)):
    for j in range(i, len(M)):
        binary.append(M[i] + "-" + M[j])
for i in range(len(M)):
    for j in range(len(S)):
        binary.append(M[i] + "-" + S[j])
ternary = []
for i in range(len(M)):
    for j in range(i, len(M)):
        for k in range(j, len(M)):
            ternary.append(M[i] + "-" + M[j] + "-" + M[k])
for i in range(len(M)):
    for j in range(i, len(M)):
        for k in range(len(S)):
            ternary.append(M[i] + "-" + M[j] + "-" + S[k])

# for cat_num in cats:
# Catalyst library
Cat_name = []
Cat_lib = []
Cat_mass_1 = []
Cat_mass_2 = []
Cat_mass_3 = []
footprint = []
for i in range(len(S)):
    for j in range(len(M)):
        for k in range(j, len(M)):
            for l in range(k, len(M)):
                Cat_name.append(M[j] + "-" + M[k] + "-" + M[l] + "-" + S[i])
                Cat_lib.append([M[j], M[k], M[l], S[i]])
                footprint.append(-10.0)
                buffer = [0.0] * len(C)
                buffer.append(1.0)
                Cat_mass_1.append(copy.deepcopy(buffer))
                Cat_mass_2.append(copy.deepcopy(buffer))
                Cat_mass_3.append(copy.deepcopy(buffer))
df_footprint = pd.DataFrame(footprint, index=Cat_name, columns=["footprint"])

# mass function initialization
n_1 = len(unary)
n_2 = len(binary)
n_3 = len(ternary)
m = len(H_value)
mass_1 = np.zeros((n_1, n_1, m + 1), dtype=float)
mass_2 = np.zeros((n_2, n_2, m + 1), dtype=float)
mass_3 = np.zeros((n_3, n_3, m + 1), dtype=float)
mass_1[..., -1] = 1.0
mass_2[..., -1] = 1.0
mass_3[..., -1] = 1.0

# data lording
df_obs = pd.read_csv(f"input/M1-M2-M3-support_{cat_num}.csv", encoding='cp932')

obs_cat = df_obs.loc[:, ["M1", "M2", "M3", "Support"]].values.tolist()
obs_per = df_obs["Measured C2 yield (%)"].values.tolist()

# C2 yield classification
C2y_class = []
for i in obs_per:
    if i >= thr:
        C2y_class.append(1)
    else:
        C2y_class.append(0)

# evidence collection
for i in range(len(obs_cat) - 1):
    C2y_1 = C2y_class[i]
    for j in range(i + 1, len(obs_cat)):
        C2y_2 = C2y_class[j]
        hypo_regular = C2y_2 - C2y_1
        hypo_reverse = C2y_1 - C2y_2
        idx_regular = H_value.index(hypo_regular)
        idx_reverse = H_value.index(hypo_reverse)
        m2_regular = [0.0] * len(H_value)
        m2_regular.append(1.0)
        m2_reverse = [0.0] * len(H_value)
        m2_reverse.append(1.0)

        # evidence recognition
        com = compare(obs_cat[i], obs_cat[j])
        if len(com[0]) == 1:  # when common part is containing 3 components
            # unary
            if weight1 != 0:
                m2_regular[idx_regular] = weight1
                m2_reverse[idx_reverse] = weight1
                m2_regular[-1] = 1.0 - weight1
                m2_reverse[-1] = 1.0 - weight1
                idx_former = unary.index(com[0][0])
                idx_latter = unary.index(com[1][0])
                m1_regular = mass_1[idx_former][idx_latter]
                m1_reverse = mass_1[idx_latter][idx_former]
                mass_1[idx_former][idx_latter] = Dempster(m1_regular, m2_regular)
                mass_1[idx_latter][idx_former] = Dempster(m1_reverse, m2_reverse)

            # binary
            if weight2 != 0:
                m2_regular[idx_regular] = weight2
                m2_reverse[idx_reverse] = weight2
                m2_regular[-1] = 1.0 - weight2
                m2_reverse[-1] = 1.0 - weight2
                for k in range(len(com[2])):
                    noncom_f = sorted([com[0][0], com[2][k]], key=sort_key)
                    noncom_l = sorted([com[1][0], com[2][k]], key=sort_key)
                    idx_former = binary.index(noncom_f[0] + "-" + noncom_f[1])
                    idx_latter = binary.index(noncom_l[0] + "-" + noncom_l[1])
                    m1_regular = mass_2[idx_former][idx_latter]
                    m1_reverse = mass_2[idx_latter][idx_former]
                    mass_2[idx_former][idx_latter] = Dempster(m1_regular, m2_regular)
                    mass_2[idx_latter][idx_former] = Dempster(m1_reverse, m2_reverse)

            # ternary
            if weight3 != 0:
                m2_regular[idx_regular] = weight3
                m2_reverse[idx_reverse] = weight3
                m2_regular[-1] = 1.0 - weight3
                m2_reverse[-1] = 1.0 - weight3
                for k in range(len(com[2]) - 1):
                    for l in range(k + 1, len(com[2])):
                        noncom_f = sorted([com[0][0], com[2][k], com[2][l]], key=sort_key)
                        noncom_l = sorted([com[1][0], com[2][k], com[2][l]], key=sort_key)
                        idx_former = ternary.index(noncom_f[0] + "-" + noncom_f[1] + "-" + noncom_f[2])
                        idx_latter = ternary.index(noncom_l[0] + "-" + noncom_l[1] + "-" + noncom_l[2])
                        m1_regular = mass_3[idx_former][idx_latter]
                        m1_reverse = mass_3[idx_latter][idx_former]
                        mass_3[idx_former][idx_latter] = Dempster(m1_regular, m2_regular)
                        mass_3[idx_latter][idx_former] = Dempster(m1_reverse, m2_reverse)

        elif len(com[0]) == 2:  # when common part is containing 2 components
            # binary
            if weight2 != 0:
                m2_regular[idx_regular] = weight2
                m2_reverse[idx_reverse] = weight2
                m2_regular[-1] = 1.0 - weight2
                m2_reverse[-1] = 1.0 - weight2
                idx_former = binary.index(com[0][0] + "-" + com[0][1])
                idx_latter = binary.index(com[1][0] + "-" + com[1][1])
                m1_regular = mass_2[idx_former][idx_latter]
                m1_reverse = mass_2[idx_latter][idx_former]
                mass_2[idx_former][idx_latter] = Dempster(m1_regular, m2_regular)
                mass_2[idx_latter][idx_former] = Dempster(m1_reverse, m2_reverse)

            # ternary
            if weight3 != 0:
                m2_regular[idx_regular] = weight3
                m2_reverse[idx_reverse] = weight3
                m2_regular[-1] = 1.0 - weight3
                m2_reverse[-1] = 1.0 - weight3
                for k in range(len(com[2])):
                    noncom_f = sorted([com[0][0], com[0][1], com[2][k]], key=sort_key)
                    noncom_l = sorted([com[1][0], com[1][1], com[2][k]], key=sort_key)
                    idx_former = ternary.index(noncom_f[0] + "-" + noncom_f[1] + "-" + noncom_f[2])
                    idx_latter = ternary.index(noncom_l[0] + "-" + noncom_l[1] + "-" + noncom_l[2])
                    m1_regular = mass_3[idx_former][idx_latter]
                    m1_reverse = mass_3[idx_latter][idx_former]
                    mass_3[idx_former][idx_latter] = Dempster(m1_regular, m2_regular)
                    mass_3[idx_latter][idx_former] = Dempster(m1_reverse, m2_reverse)

        elif len(com[0]) == 3:  # when common part is containing 1 component
            # ternary
            if weight3 != 0:
                m2_regular[idx_regular] = weight3
                m2_reverse[idx_reverse] = weight3
                m2_regular[-1] = 1.0 - weight3
                m2_reverse[-1] = 1.0 - weight3
                idx_former = ternary.index(com[0][0] + "-" + com[0][1] + "-" + com[0][2])
                idx_latter = ternary.index(com[1][0] + "-" + com[1][1] + "-" + com[1][2])
                m1_regular = mass_3[idx_former][idx_latter]
                m1_reverse = mass_3[idx_latter][idx_former]
                mass_3[idx_former][idx_latter] = Dempster(m1_regular, m2_regular)
                mass_3[idx_latter][idx_former] = Dempster(m1_reverse, m2_reverse)

# performance prediction
for j in tqdm(range(len(Cat_lib))):
    for i in range(len(obs_cat)):
        com = compare(obs_cat[i], Cat_lib[j])
        if len(com[0]) == 1:
            if weight1 != 0:
                idx_former = unary.index(com[0][0])
                idx_latter = unary.index(com[1][0])
                evidence_apply(mass_1, Cat_mass_1)

            if weight2 != 0:
                for k in range(len(com[2])):
                    noncom_f = sorted([com[0][0], com[2][k]], key=sort_key)
                    noncom_l = sorted([com[1][0], com[2][k]], key=sort_key)
                    idx_former = binary.index(noncom_f[0] + "-" + noncom_f[1])
                    idx_latter = binary.index(noncom_l[0] + "-" + noncom_l[1])
                    evidence_apply(mass_2, Cat_mass_2)

            if weight3 != 0:
                for k in range(len(com[2]) - 1):
                    for l in range(k + 1, len(com[2])):
                        noncom_f = sorted([com[0][0], com[2][k], com[2][l]], key=sort_key)
                        noncom_l = sorted([com[1][0], com[2][k], com[2][l]], key=sort_key)
                        idx_former = ternary.index(noncom_f[0] + "-" + noncom_f[1] + "-" + noncom_f[2])
                        idx_latter = ternary.index(noncom_l[0] + "-" + noncom_l[1] + "-" + noncom_l[2])
                        evidence_apply(mass_3, Cat_mass_3)

        elif len(com[0]) == 2:
            if weight2 != 0:
                idx_former = binary.index(com[0][0] + "-" + com[0][1])
                idx_latter = binary.index(com[1][0] + "-" + com[1][1])
                evidence_apply(mass_2, Cat_mass_2)

            if weight3 != 0:
                for k in range(len(com[2])):
                    noncom_f = sorted([com[0][0], com[0][1], com[2][k]], key=sort_key)
                    noncom_l = sorted([com[1][0], com[1][1], com[2][k]], key=sort_key)
                    idx_former = ternary.index(noncom_f[0] + "-" + noncom_f[1] + "-" + noncom_f[2])
                    idx_latter = ternary.index(noncom_l[0] + "-" + noncom_l[1] + "-" + noncom_l[2])
                    evidence_apply(mass_3, Cat_mass_3)

        elif len(com[0]) == 3:
            if weight3 != 0:
                idx_former = ternary.index(com[0][0] + "-" + com[0][1] + "-" + com[0][2])
                idx_latter = ternary.index(com[1][0] + "-" + com[1][1] + "-" + com[1][2])
                evidence_apply(mass_3, Cat_mass_3)

if weight1 != 0:
    df_result_1 = pd.DataFrame(Cat_mass_1, index=Cat_name, columns=["Positive_unary", "Non-positive_unary", "Uncertain_unary"])
    df_result_1 = pd.concat([df_result_1, df_footprint], axis=1)
    df_result_1.to_csv(f"output/recommender/recommender_prediction_{cat_num}_unary.csv")

if weight2 != 0:
    df_result_2 = pd.DataFrame(Cat_mass_2, index=Cat_name, columns=["Positive_binary", "Non-positive_binary", "Uncertain_binary"])
    df_result_2 = pd.concat([df_result_2, df_footprint], axis=1)
    df_result_2.to_csv(f"output/recommender/recommender_prediction_{cat_num}_binary.csv")

if weight3 != 0:
    df_result_3 = pd.DataFrame(Cat_mass_3, index=Cat_name, columns=["Positive_ternary", "Non-positive_ternary", "Uncertain_ternary"])
    df_result_3 = pd.concat([df_result_3, df_footprint], axis=1)
    df_result_3.to_csv(f"output/recommender/recommender_prediction_{cat_num}_ternary.csv")
