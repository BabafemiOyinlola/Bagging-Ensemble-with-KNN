library(ggplot2)

#read performance metric files for both datasets
iono_base_metrics <- read.csv("Ionosphere-Scratch-Different-Ks.csv", header = TRUE)
bc_base_metrics <- read.csv("Breast-Cancer-Scratch-Different-Ks.csv", header = TRUE)

iono_base_metrics <- iono_base_metrics[1:6] #delete last column
bc_base_metrics <- bc_base_metrics[1:6] #delete last column

iono_base_accuracy <- iono_base_metrics[1]
bc_base_accuracy <- bc_base_metrics[1]
k <- seq(1,20,1) #range for K parameter

df_scratch <- data.frame(
  "K" = k,
  "AccuracyIonosphere" = iono_base_accuracy,
  "AccuracyBreastCancer" = bc_base_accuracy
)

#plot the accuracy of the classifier written from scratch for both datasets across various Ks
plot_scratch <- ggplot(df_scratch, aes(x=K)) +
  geom_line(aes(y = Accuracy, colour = "#5F9EA0")) +
  geom_line(aes(y = Accuracy.1, colour = "#E1B378")) +
  ggtitle("Accuracy at different K neighbours") +
  labs(x="K", y="Accuracy")+
  scale_x_continuous(breaks=seq(1,20,1))+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(legend.position="bottom", legend.direction="horizontal",
        legend.title = element_blank())+
  scale_color_manual(labels = c("Ionosphere Data", "Breast Cancer Data"), 
                     values = c("#5F9EA0", "#E1B378"))

plot_scratch


#plot the accuracy of the library implementation of the classifier for both datasets 
#across various Ks
iono_lib_metrics <- read.csv("Ionosphere-Lib-Different-Ks.csv", header = TRUE)
bc_lib_metrics <- read.csv("Breast-Cancer-Lib-Different-Ks.csv", header = TRUE)

iono_lib_metrics <- iono_lib_metrics[1:6] #delete last column
bc_lib_metrics <- bc_lib_metrics[1:6] #delete last column

iono_lib_accuracy <- iono_lib_metrics[1]
bc_lib_accuracy <- bc_lib_metrics[1]

df_lib <- data.frame(
  "K" = k,
  "AccuracyIonosphere" = iono_lib_accuracy,
  "AccuracyBreastCancer" = bc_lib_accuracy
)


plot_lib <- ggplot(df_lib, aes(x=K)) +
  geom_line(aes(y = Accuracy, colour = "#5F9EA0")) +
  geom_line(aes(y = Accuracy.1, colour = "#E1B378")) +
  ggtitle("Accuracy at different K neighbours") +
  labs(x="K", y="Accuracy")+
  scale_x_continuous(breaks=seq(1,20,1))+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(legend.position="bottom", legend.direction="horizontal",
        legend.title = element_blank())+
  scale_color_manual(labels = c("Ionosphere Data", "Breast Cancer Data"), 
                     values = c("#5F9EA0", "#E1B378"))

plot_lib


#plot the accuracy of the classifier written from scratch for both datasets 
#across various n_estimators
#n_estimators here corresponds to the number of predictors in the ensemble
ionoK2_base_metrics <- read.csv("Ionosphere-Scratch-K=2.csv", header = TRUE)
bcK12_base_metrics <- read.csv("Breast-Cancer-Scratch-K=12.csv", header = TRUE)

ionoK2_base_metrics <- ionoK2_base_metrics[1:6] #delete last column
bcK12_base_metrics <- bcK12_base_metrics[1:6] #delete last column

ionoK2_base_accuracy <- ionoK2_base_metrics[1]
bcK12_base_accuracy <- bcK12_base_metrics[1]
n_estimator <- seq(2,20,1)

df_scratchK <- data.frame(
  "n_estimators" = n_estimator,
  "AccuracyIonosphere" = ionoK2_base_accuracy,
  "AccuracyBreastCancer" = bcK12_base_accuracy
)


plot_scratchK <- ggplot(df_scratchK, aes(x=n_estimators)) +
  geom_line(aes(y = Accuracy, colour = "#5F9EA0")) +
  geom_line(aes(y = Accuracy.1, colour = "#E1B378")) +
  ggtitle("Accuracy at K=2 / K=12 neighbours") +
  labs(x="n_estimators", y="Accuracy", color = "Data\n")+
  scale_x_continuous(breaks=seq(1,20,1))+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(legend.position="bottom", legend.direction="horizontal",
        legend.title = element_blank())+
  scale_color_manual(labels = c("Ionosphere Data", "Breast Cancer Data"), 
                     values = c("#5F9EA0", "#E1B378"))

plot_scratchK


#plot the accuracy of the library implementation of the classifier for both datasets 
#across various n_estimators
ionoK2_lib_metrics <- read.csv("Ionosphere-Lib-K=2.csv", header = TRUE)
bcK6_lib_metrics <- read.csv("Breast-Cancer-Lib-K=6.csv", header = TRUE)

ionoK2_lib_metrics <- ionoK2_lib_metrics[1:6] #delete last column
bcK6_lib_metrics <- bcK6_lib_metrics[1:6] #delete last column

ionoK2_lib_accuracy <- ionoK2_lib_metrics[1]
bcK6_lib_accuracy <- bcK6_lib_metrics[1]

df_libK <- data.frame(
  "n_estimators" = n_estimator,
  "AccuracyIonosphere" = ionoK2_lib_accuracy,
  "AccuracyBreastCancer" = bcK6_lib_accuracy
)

plot_libK <- ggplot(df_libK, aes(x=n_estimators)) +
  geom_line(aes(y = Accuracy, colour = "#5F9EA0")) +
  geom_line(aes(y = Accuracy.1, colour = "#E1B378")) +
  ggtitle("Accuracy at K=2 / K=6") +
  labs(x="n_estimators", y="Accuracy", color = "Data\n")+
  scale_x_continuous(breaks=seq(1,20,1))+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(legend.position="bottom", legend.direction="horizontal",
        legend.title = element_blank())+
  scale_color_manual(labels = c("Ionosphere Data", "Breast Cancer Data"), 
                     values = c("#5F9EA0", "#E1B378"))

plot_libK

#read file containing performace metrics obtained after 10 fold cross validation using 
#the best K and n_estimators gotten
iono_final_clf_scr <- read.csv("Ionosphere-scratch-K=2 n_est=3 CV=10.csv", header = TRUE)
iono_final_clf_lib <- read.csv("Ionosphere-Lib-K=2, n_est=8 CV=10.csv", header = TRUE)
iono_final_clf_scr <- iono_final_clf_scr[1:6]
iono_final_clf_lib <- iono_final_clf_lib[1:6]

bc_final_clf_scr <- read.csv("Breast-Cancer-scratch-K=12, n_est=5 CV=10.csv", header = TRUE)
bc_final_clf_lib <- read.csv("Breast-Cancer-Lib-K=6, n_est=7 CV=10.csv", header = TRUE)
bc_final_clf_scr <- bc_final_clf_scr[1:6]
bc_final_clf_lib <- bc_final_clf_lib[1:6]

#Statistical test
#using error as the metric
iono_final_clf_scr_df = data.frame(iono_final_clf_scr)
iono_final_clf_lib_df = data.frame(iono_final_clf_lib)

#Normality test Ionosphere
shapiro.test(iono_final_clf_scr_df$Error) #normality can be assumed
shapiro.test(iono_final_clf_lib_df$Error) #normality can be assumed

#Variance test 
var_test <- var.test(iono_final_clf_scr_df$Error,  iono_final_clf_lib_df$Error)
var_test #no variance

#t-test
t.test(iono_final_clf_scr_df$Error, iono_final_clf_lib_df$Error, var.equal = TRUE)


#calculate the average of the performace metrics after cross validation
iono_base_metrics_df <- data.frame(iono_final_clf_scr)
iono_lib_metrics_df <- data.frame(iono_final_clf_lib)

iono_av_acc_base <- mean(iono_base_metrics_df$Accuracy)
iono_av_pre_base <- mean(iono_base_metrics_df$Precision)
iono_av_rec_base <- mean(iono_base_metrics_df$Recall)
iono_av_speci_base <- mean(iono_base_metrics_df$Specificity)

iono_av_acc_lib <- mean(iono_lib_metrics_df$Accuracy)
iono_av_pre_lib <- mean(iono_lib_metrics_df$Precision)
iono_av_rec_lib <- mean(iono_lib_metrics_df$Recall)
iono_av_speci_lib <- mean(iono_lib_metrics_df$Specificity)

bc_base_metrics_df <- data.frame(bc_final_clf_scr)
bc_lib_metrics_df <- data.frame(bc_final_clf_scr)

bc_av_acc_base <- mean(bc_base_metrics_df$Accuracy)
bc_av_pre_base <- mean(bc_base_metrics_df$Precision)
bc_av_rec_base <- mean(bc_base_metrics_df$Recall)
bc_av_speci_base <- mean(bc_base_metrics_df$Specificity)

bc_av_acc_lib <- mean(bc_lib_metrics_df$Accuracy)
bc_av_pre_lib <- mean(bc_lib_metrics_df$Precision)
bc_av_rec_lib <- mean(bc_lib_metrics_df$Recall)
bc_av_speci_lib <- mean(bc_lib_metrics_df$Specificity)


#plot the ROC curve for all classifiers
library(pROC)
pred_test_Iono_scr <- read.csv("Ionosphere-Scratch-PR-ROC.csv", header = TRUE)
pred_test_Iono_scr <- pred_test_Iono_scr[1:2]

pred_test_Iono_lib <- read.csv("Ionosphere-Lib-PR-ROC.csv", header = TRUE)
pred_test_Iono_lib <- pred_test_Iono_lib[1:2]

pred_test_bc_scr <- read.csv("Breast Cancer-Scratch-PR-ROC.csv", header = TRUE)
pred_test_bc_scr <- pred_test_bc_scr[1:2]

pred_test_bc_lib <- read.csv("Breast Cancer-Lib-PR-ROC.csv", header = TRUE)
pred_test_bc_lib <- pred_test_bc_scr[1:2]

Iono_scr_roc <- roc(pred_test_Iono_scr$prediction, pred_test_Iono_scr$test)
Iono_lib_roc <- roc(pred_test_Iono_lib$prediction, pred_test_Iono_lib$test)

bc_scr_roc <- roc(pred_test_bc_scr$prediction, pred_test_bc_scr$test)
bc_lib_roc <- roc(pred_test_bc_lib$prediction, pred_test_bc_lib$test)

# Multiple curves:
ROC <- ggroc(list(Iono.Scratch=Iono_scr_roc, Iono.Lib=Iono_lib_roc, 
                  BC.Scratch=bc_scr_roc, BC.Lib=bc_lib_roc))

ROC