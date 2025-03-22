import scorer.conll18_ud_eval_EvaLatin_2022 as eval
import numpy as np

# Classical subtask evaluation
gold_ud = eval.load_conllu(open("gold_data/gold_Livius_AbVrbeCondita.conllu", "r", encoding="utf-8"))
system_ud = eval.load_conllu(open("output/test_predictions_livius.conllu", "r", encoding="utf-8"))

eval_dict = eval.evaluate(gold_ud, system_ud)

pos_scores = eval_dict["UPOS"]

print(f"Livius > UPOS - Precision: {pos_scores.precision}, Recall: {pos_scores.recall}, F1: {pos_scores.f1}, Accuracy: {pos_scores.aligned_accuracy}")

# Cross-genre subtask evaluation (Ovidius)
gold_ud = eval.load_conllu(open("gold_data/gold_Ovidius_Metamorphoseon.conllu", "r", encoding="utf-8"))
system_ud = eval.load_conllu(open("output/test_predictions_ovidius.conllu", "r", encoding="utf-8"))

eval_dict = eval.evaluate(gold_ud, system_ud)

pos_scores = eval_dict["UPOS"]

ovidius_aligned_accuracy = pos_scores.aligned_accuracy

print(f"Ovidius > UPOS - Precision: {pos_scores.precision}, Recall: {pos_scores.recall}, F1: {pos_scores.f1}, Accuracy: {pos_scores.aligned_accuracy}")

# Cross-genre subtask evaluation (Plinius)
gold_ud = eval.load_conllu(open("gold_data/gold_Plinius_NaturalisHistoria.conllu", "r", encoding="utf-8"))
system_ud = eval.load_conllu(open("output/test_predictions_plinius.conllu", "r", encoding="utf-8"))

eval_dict = eval.evaluate(gold_ud, system_ud)

pos_scores = eval_dict["UPOS"]

plinius_aligned_accuracy = pos_scores.aligned_accuracy

print(f"Plinius > UPOS - Precision: {pos_scores.precision}, Recall: {pos_scores.recall}, F1: {pos_scores.f1}, Accuracy: {pos_scores.aligned_accuracy}")

# standard deviation of the cross-genre scorers
scores = [ovidius_aligned_accuracy, plinius_aligned_accuracy]  

std_dev = np.std(scores, ddof=0)  # ddof=0 for population standard deviation, ddof=1 for sample

mean = np.mean(scores)

print(f"Standard Deviation for cross-genre task: {std_dev:.4f}")
print(f"Mean for cross-genre task: {mean:.4f}")


# Cross-time subtask evaluation
gold_ud = eval.load_conllu(open("gold_data/gold_Sabellicus_DeLatinaeLinguaeReparatione.conllu", "r", encoding="utf-8"))
system_ud = eval.load_conllu(open("output/test_predictions_sabellicus.conllu", "r", encoding="utf-8"))

eval_dict = eval.evaluate(gold_ud, system_ud)

pos_scores = eval_dict["UPOS"]

print(f"Sibellicus > UPOS - Precision: {pos_scores.precision}, Recall: {pos_scores.recall}, F1: {pos_scores.f1}, Accuracy: {pos_scores.aligned_accuracy}")
