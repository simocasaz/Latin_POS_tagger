import scorer.conll18_ud_eval_EvaLatin_2022 as eval

# Classical subtask evaluation
gold_ud = eval.load_conllu(open("gold_data/gold_Livius_AbVrbeCondita.conllu", "r", encoding="utf-8"))
system_ud = eval.load_conllu(open("output/test_predictions.conllu", "r", encoding="utf-8"))

eval_dict = eval.evaluate(gold_ud, system_ud)

pos_scores = eval_dict["UPOS"]

print(f"UPOS - Precision: {pos_scores.precision}, Recall: {pos_scores.recall}, F1: {pos_scores.f1}, Accuracy: {pos_scores.aligned_accuracy}")

# Cross-genre subtask evaluation (Ovidius)
gold_ud = eval.load_conllu(open("gold_data/gold_Ovidius_Metamorphoseon.conllu", "r", encoding="utf-8"))
system_ud = eval.load_conllu(open("output/test_predictions_ovidius.conllu", "r", encoding="utf-8"))

eval_dict = eval.evaluate(gold_ud, system_ud)

pos_scores = eval_dict["UPOS"]

print(f"UPOS - Precision: {pos_scores.precision}, Recall: {pos_scores.recall}, F1: {pos_scores.f1}, Accuracy: {pos_scores.aligned_accuracy}")

# Cross-genre subtask evaluation (Plinius)
gold_ud = eval.load_conllu(open("gold_data/gold_Plinius_NaturalisHistoria.conllu", "r", encoding="utf-8"))
system_ud = eval.load_conllu(open("output/test_predictions_plinius.conllu", "r", encoding="utf-8"))

eval_dict = eval.evaluate(gold_ud, system_ud)

pos_scores = eval_dict["UPOS"]

print(f"UPOS - Precision: {pos_scores.precision}, Recall: {pos_scores.recall}, F1: {pos_scores.f1}, Accuracy: {pos_scores.aligned_accuracy}")

# Cross-time subtask evaluation
gold_ud = eval.load_conllu(open("gold_data/gold_Sabellicus_DeLatinaeLinguaeReparatione.conllu", "r", encoding="utf-8"))
system_ud = eval.load_conllu(open("output/test_predictions_sabellicus.conllu", "r", encoding="utf-8"))

eval_dict = eval.evaluate(gold_ud, system_ud)

pos_scores = eval_dict["UPOS"]

print(f"UPOS - Precision: {pos_scores.precision}, Recall: {pos_scores.recall}, F1: {pos_scores.f1}, Accuracy: {pos_scores.aligned_accuracy}")
