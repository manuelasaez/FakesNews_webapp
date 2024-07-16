from preprocessing import Preprocess


object_preprocess = Preprocess()

object_preprocess.read_csv()
object_preprocess.convert_to_str()
object_preprocess.to_lower()
object_preprocess.text_title()
object_preprocess.remove_punct()
print(object_preprocess.train_df.iloc[0, 5])
