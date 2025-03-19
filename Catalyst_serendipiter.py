import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed

# for base model building
cv = LeaveOneOut()

# number of parallelization
p = 10

# Catalysts number for file name
cat_num = 376

# threshold for deciding performance class
thr = 13

# threshold for deciding serendipity class
thr_s = 0.2

# weight of degree of belief for one time observation
weight1 = 0.2
weight2 = 0.2
weight3 = 0.2

# [disimprove, unchange, improve]
H_value = [-1, 0, 1]
# [positive, non-positive]
C = [1, 0]

# Original library
M = ["Li", "Na", "Mg", "K", "Ca", "Ti", "V", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Sr",
     "Y", "Zr", "Mo", "Pd", "Cs", "Ba", "La", "Ce", "Nd", "Eu", "Tb", "Hf", "W", "none"]
S = ["MgO", "Al2O3", "SiO2", "CaO", "TiO2", "ZrO2", "BaO", "La2O3", "CeO2"]
sort_key = (M + S).index


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


# evidence_apply
def evidence_apply(mass_x, Cat_mass_x, idx_former, idx_latter, train_idx, test_idx, y_train):
    m1 = Cat_mass_x[test_idx]
    m2_origin = mass_x[idx_former][idx_latter]
    if m2_origin[-1] == 1.0:
        return
    m2 = [0.0] * len(C)
    m2.append(1.0)
    for m in range(len(H_value)):
        if (y_train[train_idx] == 1 and H_value[m] in [1, 0]):
            predicted_C2y = 1
        elif (y_train[train_idx] == 0 and H_value[m] in [-1, 0]):
            predicted_C2y = 0
        else:
            continue
        idx_C2y = C.index(predicted_C2y)
        m2[idx_C2y] += m2_origin[m]
    m2[-1] = 1.0 - sum(m2[:-1])
    updated_mass = Dempster(m1, m2)
    Cat_mass_x[test_idx][:] = updated_mass
    return


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

Cat_lib = []
OneHot = []
Cat_name = []
for i in range(len(S)):
    for j in range(len(M)):
        for k in range(j, len(M)):
            for l in range(k, len(M)):
                Cat_name.append(M[j] + "-" + M[k] + "-" + M[l] + "-" + S[i])
                Cat_lib.append([M[j], M[k], M[l], S[i]])
                buffer = []
                for m in unary:
                    buffer.append([M[j], M[k], M[l], S[i]].count(m))
                OneHot.append(buffer)
df_cat = pd.DataFrame(Cat_lib, columns=["M1", "M2", "M3", "Support"], index=Cat_name)
df_OneHot = pd.DataFrame(OneHot, columns=unary, index=Cat_name)

c_unary = ["Positive_unary", "Non-positive_unary", "Uncertain_unary"]
c_binary = ["Positive_binary", "Non-positive_binary", "Uncertain_binary"]
c_ternary = ["Positive_ternary", "Non-positive_ternary", "Uncertain_ternary"]

df_obs = pd.read_csv("input/M1-M2-M3-support_" + str(cat_num) + ".csv", encoding='cp932')

X = df_obs.loc[:, ["M1", "M2", "M3", "Support"]].values
obs_per_origin = df_obs["Measured C2 yield (%)"].values

# mass function initialization
n_1 = len(unary)
n_2 = len(binary)
n_3 = len(ternary)
m = len(H_value)
mass_1_origin = np.zeros((n_1, n_1, m + 1), dtype=float)
mass_2_origin = np.zeros((n_2, n_2, m + 1), dtype=float)
mass_3_origin = np.zeros((n_3, n_3, m + 1), dtype=float)
mass_1_origin[..., -1] = 1.0
mass_2_origin[..., -1] = 1.0
mass_3_origin[..., -1] = 1.0

C2y_class = []
name = []
for i in range(len(obs_per_origin)):
    if obs_per_origin[i] >= thr:
        C2y_class.append(1)
    else:
        C2y_class.append(0)
    cat = X[i][0] + "-" + X[i][1] + "-" + X[i][2] + "-" + X[i][3]
    name.append(cat)
y = np.array(C2y_class)


def base_models(train_index, test_index):
    X_train, X_test = X[train_index].tolist(), X[test_index].tolist()
    y_train, y_test = y[train_index].tolist(), y[test_index].tolist()

    Cat_name = []
    Cat_lib = []
    Cat_mass_1 = []
    Cat_mass_2 = []
    Cat_mass_3 = []
    for idx in range(len(X_test)):
        cat_str = X_test[idx][0] + "-" + X_test[idx][1] + "-" + X_test[idx][2] + "-" + X_test[idx][3]
        Cat_name.append(cat_str)
        Cat_lib.append([X_test[idx][0], X_test[idx][1], X_test[idx][2], X_test[idx][3]])
        buffer = [0.0] * len(C)
        buffer.append(1.0)
        Cat_mass_1.append(copy.deepcopy(buffer))
        Cat_mass_2.append(copy.deepcopy(buffer))
        Cat_mass_3.append(copy.deepcopy(buffer))
    df_footprint = pd.DataFrame(y_test, index=Cat_name, columns=["footprint"])

    mass_1 = copy.deepcopy(mass_1_origin)
    mass_2 = copy.deepcopy(mass_2_origin)
    mass_3 = copy.deepcopy(mass_3_origin)

    # evidence collection
    for i in range(len(X_train) - 1):
        for j in range(i + 1, len(X_train)):
            hypo_regular = y_train[j] - y_train[i]
            hypo_reverse = y_train[i] - y_train[j]
            idx_regular = H_value.index(hypo_regular)
            idx_reverse = H_value.index(hypo_reverse)
            m2_regular = [0.0] * len(H_value)
            m2_regular.append(1.0)
            m2_reverse = [0.0] * len(H_value)
            m2_reverse.append(1.0)

            com = compare(X_train[i], X_train[j])
            if len(com[0]) == 1:  # 共通成分が3個（差分1個）
                # unary branch
                m2_regular[idx_regular] = weight1
                m2_reverse[idx_reverse] = weight1
                m2_regular[-1] = 1.0 - weight1
                m2_reverse[-1] = 1.0 - weight1
                idx_former = unary.index(com[0][0])
                idx_latter = unary.index(com[1][0])
                mass_1[idx_former][idx_latter] = Dempster(mass_1[idx_former][idx_latter], m2_regular)
                mass_1[idx_latter][idx_former] = Dempster(mass_1[idx_latter][idx_former], m2_reverse)

                # binary branch
                m2_regular[idx_regular] = weight2
                m2_reverse[idx_reverse] = weight2
                m2_regular[-1] = 1.0 - weight2
                m2_reverse[-1] = 1.0 - weight2
                for k in range(len(com[2])):
                    noncom_f = sorted([com[0][0], com[2][k]], key=sort_key)
                    noncom_l = sorted([com[1][0], com[2][k]], key=sort_key)
                    idx_former = binary.index(noncom_f[0] + "-" + noncom_f[1])
                    idx_latter = binary.index(noncom_l[0] + "-" + noncom_l[1])
                    mass_2[idx_former][idx_latter] = Dempster(mass_2[idx_former][idx_latter], m2_regular)
                    mass_2[idx_latter][idx_former] = Dempster(mass_2[idx_latter][idx_former], m2_reverse)

                # ternary branch
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
                        mass_3[idx_former][idx_latter] = Dempster(mass_3[idx_former][idx_latter], m2_regular)
                        mass_3[idx_latter][idx_former] = Dempster(mass_3[idx_latter][idx_former], m2_reverse)

            elif len(com[0]) == 2:  # 共通成分が2個
                # binary branch
                m2_regular[idx_regular] = weight2
                m2_reverse[idx_reverse] = weight2
                m2_regular[-1] = 1.0 - weight2
                m2_reverse[-1] = 1.0 - weight2
                idx_former = binary.index(com[0][0] + "-" + com[0][1])
                idx_latter = binary.index(com[1][0] + "-" + com[1][1])
                mass_2[idx_former][idx_latter] = Dempster(mass_2[idx_former][idx_latter], m2_regular)
                mass_2[idx_latter][idx_former] = Dempster(mass_2[idx_latter][idx_former], m2_reverse)

                # ternary branch
                m2_regular[idx_regular] = weight3
                m2_reverse[idx_reverse] = weight3
                m2_regular[-1] = 1.0 - weight3
                m2_reverse[-1] = 1.0 - weight3
                for k in range(len(com[2])):
                    noncom_f = sorted([com[0][0], com[0][1], com[2][k]], key=sort_key)
                    noncom_l = sorted([com[1][0], com[1][1], com[2][k]], key=sort_key)
                    idx_former = ternary.index(noncom_f[0] + "-" + noncom_f[1] + "-" + noncom_f[2])
                    idx_latter = ternary.index(noncom_l[0] + "-" + noncom_l[1] + "-" + noncom_l[2])
                    mass_3[idx_former][idx_latter] = Dempster(mass_3[idx_former][idx_latter], m2_regular)
                    mass_3[idx_latter][idx_former] = Dempster(mass_3[idx_latter][idx_former], m2_reverse)

            elif len(com[0]) == 3:  # 共通成分が1個
                # ternary branch
                m2_regular[idx_regular] = weight3
                m2_reverse[idx_reverse] = weight3
                m2_regular[-1] = 1.0 - weight3
                m2_reverse[-1] = 1.0 - weight3
                idx_former = ternary.index(com[0][0] + "-" + com[0][1] + "-" + com[0][2])
                idx_latter = ternary.index(com[1][0] + "-" + com[1][1] + "-" + com[1][2])
                mass_3[idx_former][idx_latter] = Dempster(mass_3[idx_former][idx_latter], m2_regular)
                mass_3[idx_latter][idx_former] = Dempster(mass_3[idx_latter][idx_former], m2_reverse)

    # performance prediction
    for j in range(len(X_test)):
        for i in range(len(X_train)):
            com = compare(X_train[i], Cat_lib[j])
            if len(com[0]) == 1:
                # unary branch
                idx_former = unary.index(com[0][0])
                idx_latter = unary.index(com[1][0])
                evidence_apply(mass_1, Cat_mass_1, idx_former, idx_latter, i, j, y_train)
                # binary branch
                for k in range(len(com[2])):
                    noncom_f = sorted([com[0][0], com[2][k]], key=sort_key)
                    noncom_l = sorted([com[1][0], com[2][k]], key=sort_key)
                    idx_former = binary.index(noncom_f[0] + "-" + noncom_f[1])
                    idx_latter = binary.index(noncom_l[0] + "-" + noncom_l[1])
                    evidence_apply(mass_2, Cat_mass_2, idx_former, idx_latter, i, j, y_train)
                # ternary branch
                for k in range(len(com[2]) - 1):
                    for l in range(k + 1, len(com[2])):
                        noncom_f = sorted([com[0][0], com[2][k], com[2][l]], key=sort_key)
                        noncom_l = sorted([com[1][0], com[2][k], com[2][l]], key=sort_key)
                        idx_former = ternary.index(noncom_f[0] + "-" + noncom_f[1] + "-" + noncom_f[2])
                        idx_latter = ternary.index(noncom_l[0] + "-" + noncom_l[1] + "-" + noncom_l[2])
                        evidence_apply(mass_3, Cat_mass_3, idx_former, idx_latter, i, j, y_train)
            elif len(com[0]) == 2:
                # binary branch
                idx_former = binary.index(com[0][0] + "-" + com[0][1])
                idx_latter = binary.index(com[1][0] + "-" + com[1][1])
                evidence_apply(mass_2, Cat_mass_2, idx_former, idx_latter, i, j, y_train)
                for k in range(len(com[2])):
                    noncom_f = sorted([com[0][0], com[0][1], com[2][k]], key=sort_key)
                    noncom_l = sorted([com[1][0], com[1][1], com[2][k]], key=sort_key)
                    idx_former = ternary.index(noncom_f[0] + "-" + noncom_f[1] + "-" + noncom_f[2])
                    idx_latter = ternary.index(noncom_l[0] + "-" + noncom_l[1] + "-" + noncom_l[2])
                    evidence_apply(mass_3, Cat_mass_3, idx_former, idx_latter, i, j, y_train)
            elif len(com[0]) == 3:
                # ternary branch
                idx_former = ternary.index(com[0][0] + "-" + com[0][1] + "-" + com[0][2])
                idx_latter = ternary.index(com[1][0] + "-" + com[1][1] + "-" + com[1][2])
                evidence_apply(mass_3, Cat_mass_3, idx_former, idx_latter, i, j, y_train)

    df_result_1 = pd.DataFrame(Cat_mass_1, index=Cat_name, columns=c_unary)
    df_result_2 = pd.DataFrame(Cat_mass_2, index=Cat_name, columns=c_binary)
    df_result_3 = pd.DataFrame(Cat_mass_3, index=Cat_name, columns=c_ternary)

    # One-Hot encoding for training
    OneHot_1 = []
    for j0 in range(len(X_train)):
        buffer_1 = []
        for k0 in unary:
            buffer_1.append(X_train[j0].count(k0))
        OneHot_1.append(buffer_1)
    X_train_unary = pd.DataFrame(OneHot_1, columns=unary)

    # One-Hot encoding for test
    idx_list = []
    OneHot_1 = []
    for j0 in range(len(X_test)):
        buffer_1 = []
        for k0 in unary:
            buffer_1.append(X_test[j0].count(k0))
        OneHot_1.append(buffer_1)
        cat = X_test[j0][0] + "-" + X_test[j0][1] + "-" + X_test[j0][2] + "-" + X_test[j0][3]
        df_footprint.loc[cat] = y_test[j0]
        idx_list.append(cat)
    X_test_unary = pd.DataFrame(OneHot_1, index=idx_list, columns=unary)

    df_result = pd.concat([df_result_1, df_result_2, df_result_3, df_footprint], axis=1)

    df_buffer = pd.DataFrame()
    for v in range(len(X_test)):
        cat = X_test[v][0] + "-" + X_test[v][1] + "-" + X_test[v][2] + "-" + X_test[v][3]
        df_buffer = pd.concat([df_buffer, df_result.loc[cat]], axis=1)

    # SVM
    clf = SVC(C=1, gamma="scale", kernel="rbf")
    clf.fit(X_train_unary, y_train)
    y_validate = clf.predict(X_test_unary)
    df_buffer.loc["SVM"] = y_validate

    # RandomForest
    clf = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2,
                                 n_estimators=200, random_state=42)
    clf.fit(X_train_unary, y_train)
    y_validate = clf.predict(X_test_unary)
    df_buffer.loc["RF"] = y_validate

    df_buffer = pd.concat([X_test_unary.T, df_buffer])

    result = df_buffer.values.flatten().tolist()
    return result


# base model building
results = Parallel(n_jobs=p)(
    delayed(base_models)(train_index, test_index)
    for train_index, test_index in tqdm(cv.split(X))
)

feature_names = unary + c_unary + c_binary + c_ternary + ['y_actual', 'SVM', 'RF']
df_results = pd.DataFrame(results, index=name, columns=feature_names)

df_results = df_results[df_results["Non-positive_binary"] - df_results["Positive_binary"] > thr_s]
df_results.to_csv("output/serendipiter/serendipiter_descriptors_" + str(cat_num) + ".csv")

# serendipiter training
y_train_s = df_results["y_actual"]
X_train_unary = df_results[unary]
X_train_s = df_results.drop(unary + ["y_actual"], axis=1)

serendipiter = SVC(C=1, class_weight="balanced", gamma="scale", kernel='poly', random_state=42)
serendipiter.fit(X_train_s, y_train_s)

# lord results of inverse analysis of recommenders
df_unary = pd.read_csv(f"output/recommender/recommender_prediction_{cat_num}_unary.csv", index_col=0)
df_binary = pd.read_csv(f"output/recommender/recommender_prediction_{cat_num}_binary.csv", index_col=0)
df_ternary = pd.read_csv(f"output/recommender/recommender_prediction_{cat_num}_ternary.csv", index_col=0)

df_test = pd.concat(
    [df_unary.iloc[:, 0:-1],
     df_binary.iloc[:, 0:-1],
     df_ternary.iloc[:, 0:-1]],
    axis=1
)

clf = SVC(C=1, gamma="scale", kernel="rbf")
clf.fit(X_train_unary, y_train_s)
y_pred = clf.predict(df_OneHot)
df_test["SVM"] = y_pred

clf = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2,
                             n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train_unary, y_train_s)
y_pred = clf.predict(df_OneHot)
df_test["RF"] = y_pred

df_test = df_test.reindex(columns=X_train_s.columns)

y_pred = serendipiter.predict(df_test)
df_test['prediction'] = y_pred

df_test = pd.concat([df_cat, df_test], axis=1)

df_test.to_csv("output/serendipiter/serendipiter_prediction_" + str(cat_num) + ".csv")


