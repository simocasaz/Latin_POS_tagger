import conll18_ud_eval_EvaLatin_2022 as eval

gold_ud = eval.load_conllu(open("scorer/gold_Livius_AbVrbeCondita.conllu", "r", encoding="utf-8"))
system_ud = eval.load_conllu(open("scorer/test_predictions.conllu", "r", encoding="utf-8"))

eval_dict = eval.evaluate(gold_ud, system_ud)

pos_scores = eval_dict["UPOS"]

print(print(f"UPOS - Precision: {pos_scores.precision}, Recall: {pos_scores.recall}, F1: {pos_scores.f1}, Accuracy: {pos_scores.aligned_accuracy}"))