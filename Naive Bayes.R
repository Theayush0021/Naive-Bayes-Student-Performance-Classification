
library(e1071)

# 1. Load dataset
students <- read.csv("StudentsPerformance.csv")

# 2. Create average score and result
students$average_score <- (
  students$math.score +
    students$reading.score +
    students$writing.score
) / 3

students$result <- factor(
  ifelse(students$average_score >= 60,
         "Pass",
         "Needs Improvement")
)

# 3. Split data into training and testing
set.seed(123)

index <- sample(
  seq_len(nrow(students)),
  floor(0.8 * nrow(students))
)

train_data <- students[index, ]
test_data <- students[-index, ]

# 4. Train Naive Bayes model
nb_model <- naiveBayes(
  result ~ gender +
    race.ethnicity +
    parental.level.of.education +
    lunch +
    test.preparation.course,
  data = train_data
)

# 5. Make predictions
predictions <- predict(nb_model, test_data)

# 6. Confusion matrix
cm <- table(
  Actual = test_data$result,
  Predicted = predictions
)

print(cm)

# 7. Calculate performance metrics
TN <- cm["Needs Improvement", "Needs Improvement"]
FP <- cm["Needs Improvement", "Pass"]
FN <- cm["Pass", "Needs Improvement"]
TP <- cm["Pass", "Pass"]

accuracy <- (TP + TN) / sum(cm) * 100
precision <- TP / (TP + FP) * 100
recall <- TP / (TP + FN) * 100
f1_score <- 2 * precision * recall / (precision + recall)

cat("Accuracy:", round(accuracy, 2), "%\n")
cat("Precision:", round(precision, 2), "%\n")
cat("Recall:", round(recall, 2), "%\n")
cat("F1 Score:", round(f1_score, 2), "%\n")

# 8. Save confusion matrix
write.csv(
  as.data.frame.matrix(cm),
  "naive_bayes_confusion_matrix.csv"
)

# 9. Save performance metrics
metrics <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score"),
  Value = round(c(accuracy, precision, recall, f1_score), 2)
)

write.csv(
  metrics,
  "naive_bayes_metrics.csv",
  row.names = FALSE
)

# 10. Save predictions
results <- data.frame(
  Actual = test_data$result,
  Predicted = predictions
)

write.csv(
  results,
  "naive_bayes_predictions.csv",
  row.names = FALSE
)

# 11. Save confusion matrix plot
png("naive_bayes_confusion_matrix.png",
    width = 800, height = 600)

barplot(
  cm,
  beside = TRUE,
  col = c("skyblue", "orange"),
  main = "Naive Bayes Confusion Matrix",
  xlab = "Predicted Class",
  ylab = "Number of Students",
  legend.text = rownames(cm)
)

dev.off()

cat("All project files saved successfully!\n")
