epochs = 100
batch_size = 4  # Should be very small to ensure training convergence.
num_batches = 100
learning_rate = 1e-6  # Should be also *very* small. High rates make training stagnate.
bert_output_dim = 768
csv_rows_limit = batch_size * num_batches

model_name = "deep_judge"
article_csv_selector = "text"
label_csv_selector = "label"
title_csv_selector = "title"
