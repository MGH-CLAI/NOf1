# ====================== N-of-1 Simulation: Full Script =======================
# Thesis: Average (population-optimized) models look fine on global metrics
#         but fail special/rare cases. A multi-agent system with a rare
#         specialist (using a rare-only measurement x3) + stacking wins big.
# ============================================================================

set.seed(2025)

# Packages --------------------------------------------------------------------
pkgs <- c("ggplot2","dplyr","tidyr","pROC","MASS","randomForest",
          "gridExtra","patchwork","cowplot")
inst <- rownames(installed.packages())
if(any(!pkgs %in% inst)) install.packages(pkgs[!pkgs %in% inst], repos="https://cloud.r-project.org")
invisible(lapply(pkgs, library, character.only=TRUE))

# Helpers ---------------------------------------------------------------------
logit   <- function(z) 1/(1+exp(-z))
auc_num <- function(prob, y) as.numeric(pROC::roc(y, prob, quiet=TRUE)$auc)
acc_at  <- function(p, y, thr=0.5) mean((p>thr)==y)

# Local density proxy: inverse k-th NN distance
knn_density <- function(Xtrain, X, k=25){
  Xt <- as.matrix(Xtrain); Xq <- as.matrix(X)
  apply(Xq, 1, function(z){
    d <- sqrt(rowSums((Xt - matrix(z, nrow(Xt), ncol(Xt), byrow=TRUE))^2))
    sort_d <- sort(d, partial=k)[k]
    1/(sort_d + 1e-6)
  })
}

# Stratified split by cluster (60/20/20)
stratified_split <- function(df, cluster_col="cluster", p_train=0.6, p_val=0.2){
  levs <- unique(df[[cluster_col]])
  tr <- list(); va <- list(); te <- list()
  for(cl in levs){
    ix <- which(df[[cluster_col]]==cl)
    m  <- length(ix)
    set.seed(7 + which(levs==cl))
    tr_ix <- sample(ix, round(p_train*m))
    va_ix <- sample(setdiff(ix, tr_ix), round(p_val*m))
    te_ix <- setdiff(ix, c(tr_ix, va_ix))
    tr[[cl]] <- df[tr_ix, , drop=FALSE]
    va[[cl]] <- df[va_ix, , drop=FALSE]
    te[[cl]] <- df[te_ix, , drop=FALSE]
  }
  list(train = dplyr::bind_rows(tr),
       val   = dplyr::bind_rows(va),
       test  = dplyr::bind_rows(te))
}

# =============================== 1) Data =====================================
# Majority clusters (A,B,C) use x1,x2; Rare cluster (D) has an extra signal x3
# that is NOT available for A/B/C. Outcome in D depends strongly on x3.
nA <- 6000; nB <- 4000; nC <- 2000; nD <- 400
make_cluster <- function(n, mu, Sigma) MASS::mvrnorm(n, mu=mu, Sigma=Sigma)

Sigma_major <- matrix(c(1.0, 0.25, 0.25, 1.0), 2, 2)
A <- make_cluster(nA, c(-1.2,  1.1), Sigma_major)
B <- make_cluster(nB, c( 1.4,  1.0), Sigma_major)
C <- make_cluster(nC, c( 0.1, -1.3), Sigma_major)

Sigma_rare  <- matrix(c(0.35, -0.2, -0.2, 0.5), 2, 2)
D <- make_cluster(nD, c(2.8, -2.6), Sigma_rare)

X12 <- rbind(A,B,C,D)
cluster_id <- c(rep("A",nA), rep("B",nB), rep("C",nC), rep("D",nD))
colnames(X12) <- c("x1","x2")

# x3: only measured in D; NA elsewhere
x3 <- rep(NA_real_, nrow(X12))
x3[cluster_id=="D"] <- rnorm(sum(cluster_id=="D"), mean=0, sd=1)

df <- data.frame(x1=X12[,1], x2=X12[,2], x3=x3, cluster=cluster_id, stringsAsFactors=FALSE)

# Labels: majority logistic on x1,x2; rare cluster uses x3 (strong)
eta_maj <- with(df, 0.8 + 1.2*x1 - 0.9*x2 + 0.35*(x1*x2))
p_maj   <- logit(eta_maj)
beta3   <- 3.5
p_rare  <- with(df, logit(beta3 * x3))                 # meaningful only where x3 observed
noise   <- ifelse(df$cluster=="D", 0, 0.03)
p_raw   <- ifelse(df$cluster=="D", p_rare, p_maj)
p       <- pmin(pmax(p_raw*(1-noise) + (1-p_raw)*noise, 1e-5), 1-1e-5)
df$y    <- rbinom(nrow(df), 1, p)

# =============================== 2) Split ====================================
spl   <- stratified_split(df, cluster_col="cluster", p_train=0.6, p_val=0.2)
train <- spl$train; val <- spl$val; test <- spl$test

# =================== 3) Monolithic "average" baseline ========================
# Cannot use x3 because it's missing for most patients; combine GLM + RF on x1,x2.
monolog <- glm(y ~ poly(x1,2,raw=TRUE) + poly(x2,2,raw=TRUE) + x1:x2,
               data=train, family=binomial())
monorf  <- randomForest(as.factor(y) ~ x1 + x2, data=train, ntree=600)

pred_mono_val  <- 0.5*predict(monolog, newdata=val,  type="response") +
  0.5*as.numeric(predict(monorf,  newdata=val, type="prob")[,2])
pred_mono_test <- 0.5*predict(monolog, newdata=test, type="response") +
  0.5*as.numeric(predict(monorf,  newdata=test, type="prob")[,2])

# ======================= 4) Multi-agent system ===============================
# 4.1 Regions by k-means on x1,x2 (rare handled separately via x3)
set.seed(11)
K <- 3
km <- kmeans(train[,c("x1","x2")], centers=K, nstart=25)
train$agent <- paste0("agent", km$cluster)
agents  <- sort(unique(train$agent))
centers <- km$centers

# 4.2 Fit agent specialists; choose GLM vs RF via 3-fold AUC (x1,x2 only)
fit_agent <- function(d){
  set.seed(13)
  fold <- sample(1:3, nrow(d), replace=TRUE)
  
  # GLM
  pr_glm <- rep(NA_real_, nrow(d))
  for(ff in 1:3){
    fit <- glm(y ~ poly(x1,2,raw=TRUE)+poly(x2,2,raw=TRUE)+x1:x2,
               data=d[fold!=ff,], family=binomial())
    pr_glm[fold==ff] <- predict(fit, newdata=d[fold==ff,], type="response")
  }
  auc_glm <- auc_num(pr_glm, d$y)
  m_glm   <- glm(y ~ poly(x1,2,raw=TRUE)+poly(x2,2,raw=TRUE)+x1:x2, data=d, family=binomial())
  
  # RF
  pr_rf <- rep(NA_real_, nrow(d))
  for(ff in 1:3){
    fit <- randomForest(as.factor(y) ~ x1 + x2, data=d[fold!=ff,], ntree=400)
    pr_rf[fold==ff] <- as.numeric(predict(fit, d[fold==ff,], type="prob")[,2])
  }
  auc_rf <- auc_num(pr_rf, d$y)
  m_rf   <- randomForest(as.factor(y) ~ x1 + x2, data=d, ntree=600)
  
  if(auc_rf > auc_glm){
    list(model=m_rf, type="rf", auc=auc_rf)
  } else {
    list(model=m_glm, type="glm", auc=auc_glm)
  }
}
agent_models <- lapply(agents, function(a) fit_agent(train[train$agent==a,]))
names(agent_models) <- agents

# 4.3 Rare specialist that CAN use x3
rare_train <- train[!is.na(train$x3), ]
if(nrow(rare_train) < 50) stop("Not enough rare cases with x3 to train specialist.")
rare_model <- glm(y ~ x3, data=rare_train, family=binomial())

# 4.4 Utilities for agent predictions + softmax proximity weights
softmax <- function(v) exp(v - max(v))/sum(exp(v - max(v)))
agent_predict <- function(newdf){
  X <- as.matrix(newdf[,c("x1","x2")])
  D <- sapply(1:nrow(centers), function(j){
    rowSums((X - matrix(centers[j,], nrow(X), 2, byrow=TRUE))^2)
  })
  prox <- 1/(D + 1e-6)
  Wprox <- t(apply(prox, 1, softmax))
  P <- sapply(agents, function(a){
    obj <- agent_models[[a]]
    mdl <- obj$model
    if(obj$type=="rf"){
      as.numeric(predict(mdl, newdf, type="prob")[,2])
    } else {
      predict(mdl, newdf, type="response")
    }
  })
  if(is.null(dim(P))) P <- matrix(P, ncol=length(agents))
  list(P=P, Wprox=Wprox)
}

# 4.5 Density for reporting and meta features
Xtrain_mat <- as.matrix(train[,c("x1","x2")])
train$density <- knn_density(Xtrain_mat, train[,c("x1","x2")], k=25)
val$density   <- knn_density(Xtrain_mat, val[,c("x1","x2")],   k=25)
test$density  <- knn_density(Xtrain_mat, test[,c("x1","x2")],  k=25)
dens_thr <- quantile(train$density, 0.08)

# 4.6 Stacked meta-learner on VAL (uses agent probs + density + disagreement)
val_ag   <- agent_predict(val)
P_val    <- val_ag$P
colnames(P_val) <- paste0("p_", agents)
val$disagree    <- apply(P_val, 1, sd)
meta_df <- cbind(val[,c("y","density","disagree")], as.data.frame(P_val))
w_tail  <- 1 + 3*as.numeric(val$density < dens_thr)    # upweight tails
meta_fit <- glm(y ~ ., data=meta_df, family=binomial(), weights=w_tail)

# 4.7 Final multi-agent on TEST:
#     use rare specialist when x3 is present; otherwise use stacked meta-learner.
ap_test <- agent_predict(test)
P_test  <- ap_test$P
colnames(P_test) <- paste0("p_", agents)
test$disagree <- apply(P_test, 1, sd)
meta_test <- cbind(test[,c("density","disagree")], as.data.frame(P_test))
p_stack_test <- predict(meta_fit, newdata=meta_test, type="response")

p_multi <- ifelse(!is.na(test$x3),
                  predict(rare_model, newdata=test, type="response"),
                  p_stack_test)

# ============================ 5) Evaluation ===================================
yte <- test$y

# Overall
auc_mono  <- auc_num(pred_mono_test, yte)
auc_multi <- auc_num(p_multi,        yte)
acc_mono  <- acc_at(pred_mono_test, yte)
acc_multi <- acc_at(p_multi,        yte)

# Tail = top 12% by Mahalanobis distance on x1,x2
S  <- cov(train[,c("x1","x2")]); mu <- colMeans(train[,c("x1","x2")])
md <- mahalanobis(test[,c("x1","x2")], center=mu, cov=S)
tail_ix <- order(md, decreasing=TRUE)[1:ceiling(0.12*nrow(test))]
y_tail  <- yte[tail_ix]

auc_mono_tail  <- auc_num(pred_mono_test[tail_ix], y_tail)
auc_multi_tail <- auc_num(p_multi[tail_ix],        y_tail)
acc_mono_tail  <- acc_at(pred_mono_test[tail_ix],  y_tail)
acc_multi_tail <- acc_at(p_multi[tail_ix],         y_tail)

# Per-cluster metrics
per_cluster <- test %>%
  mutate(p_mono = pred_mono_test, p_multi = p_multi) %>%
  group_by(cluster) %>%
  summarise(
    n        = dplyr::n(),
    AUC_mono = auc_num(p_mono,  y),
    AUC_multi= auc_num(p_multi, y),
    Acc_mono = acc_at(p_mono,  y),
    Acc_multi= acc_at(p_multi, y),
    .groups  = "drop"
  )

cat("\n================= SUMMARY =================\n")
cat(sprintf("Monolith:    AUC=%.3f  Acc=%.3f | Tail: AUC=%.3f  Acc=%.3f\n",
            auc_mono, acc_mono, auc_mono_tail, acc_mono_tail))
cat(sprintf("Multi-agent: AUC=%.3f  Acc=%.3f | Tail: AUC=%.3f  Acc=%.3f\n",
            auc_multi, acc_multi, auc_multi_tail, acc_multi_tail))
cat("\nPer-cluster (test):\n"); print(as.data.frame(per_cluster))

# ===================== 6) Stats & Results Table ==============================
# DeLong tests and CIs
roc_mono_all   <- pROC::roc(yte, pred_mono_test, quiet=TRUE)
roc_multi_all  <- pROC::roc(yte, p_multi,        quiet=TRUE)
test_all       <- pROC::roc.test(roc_mono_all,  roc_multi_all,  method="delong")

roc_mono_tail  <- pROC::roc(y_tail, pred_mono_test[tail_ix], quiet=TRUE)
roc_multi_tail <- pROC::roc(y_tail, p_multi[tail_ix],        quiet=TRUE)
test_tail      <- pROC::roc.test(roc_mono_tail, roc_multi_tail, method="delong")

ixD <- which(test$cluster=="D")
roc_mono_D  <- pROC::roc(test$y[ixD], pred_mono_test[ixD], quiet=TRUE)
roc_multi_D <- pROC::roc(test$y[ixD], p_multi[ixD],        quiet=TRUE)
test_D      <- pROC::roc.test(roc_mono_D, roc_multi_D, method="delong")

ci_all_mono   <- pROC::ci.auc(roc_mono_all)
ci_all_multi  <- pROC::ci.auc(roc_multi_all)
ci_tail_mono  <- pROC::ci.auc(roc_mono_tail)
ci_tail_multi <- pROC::ci.auc(roc_multi_tail)
ci_D_mono     <- pROC::ci.auc(roc_mono_D)
ci_D_multi    <- pROC::ci.auc(roc_multi_D)
fmt_ci <- function(ci) sprintf("[%.3f, %.3f]", ci[1], ci[3])

accD_mono  <- mean((pred_mono_test[ixD]>0.5)==test$y[ixD])
accD_multi <- mean((p_multi[ixD]>0.5)==test$y[ixD])

results_tbl <- data.frame(
  Setting       = c("Overall","Tail (top 12%)","Rare cluster D"),
  AUC_Monolith  = c(as.numeric(roc_mono_all$auc),
                    as.numeric(roc_mono_tail$auc),
                    as.numeric(roc_mono_D$auc)),
  AUC_Multi     = c(as.numeric(roc_multi_all$auc),
                    as.numeric(roc_multi_tail$auc),
                    as.numeric(roc_multi_D$auc)),
  AUC_Delta     = c(as.numeric(roc_multi_all$auc - roc_mono_all$auc),
                    as.numeric(roc_multi_tail$auc - roc_mono_tail$auc),
                    as.numeric(roc_multi_D$auc - roc_mono_D$auc)),
  AUC_CI_Mono   = c(fmt_ci(ci_all_mono),  fmt_ci(ci_tail_mono),  fmt_ci(ci_D_mono)),
  AUC_CI_Multi  = c(fmt_ci(ci_all_multi), fmt_ci(ci_tail_multi), fmt_ci(ci_D_multi)),
  AUC_p_Delong  = c(test_all$p.value, test_tail$p.value, test_D$p.value),
  Acc_Monolith  = c(acc_mono, acc_mono_tail, accD_mono),
  Acc_Multi     = c(acc_multi, acc_multi_tail, accD_multi)
)
results_tbl$Acc_Delta <- results_tbl$Acc_Multi - results_tbl$Acc_Monolith

# >>> FIX: round only numeric columns <<<
num_cols <- sapply(results_tbl, is.numeric)
results_tbl_fmt <- results_tbl
results_tbl_fmt[num_cols] <- lapply(results_tbl_fmt[num_cols], function(x) round(x, 3))

cat("\n==== Publication Table ====\n"); print(results_tbl_fmt, row.names=FALSE)

# ===================== 7) Camera-ready Figures ===============================
# ECE
ece <- function(p, y, bins=20){
  br <- quantile(p, probs=seq(0,1,length.out=bins+1), na.rm=TRUE)
  br[1] <- min(br[1],0); br[length(br)] <- max(br[length(br)],1)
  b <- cut(p, breaks=unique(br), include.lowest=TRUE)
  d <- aggregate(list(pred=p, obs=y), by=list(bin=b), FUN=mean, na.rm=TRUE)
  N <- as.numeric(table(b)[as.character(d$bin)])
  sum(abs(d$pred - d$obs) * N) / sum(N)
}

df_test <- test
df_test$p_mono  <- pred_mono_test
df_test$p_multi <- p_multi
S_test  <- cov(train[,c("x1","x2")]); mu_test <- colMeans(train[,c("x1","x2")])
df_test$md      <- mahalanobis(df_test[,c("x1","x2")], center=mu_test, cov=S_test)
tail_cut        <- sort(df_test$md, decreasing=TRUE)[ceiling(0.12*nrow(df_test))]
df_test$is_tail <- df_test$md >= tail_cut

ece_mono_tail  <- ece(df_test$p_mono [df_test$is_tail], df_test$y[df_test$is_tail])
ece_multi_tail <- ece(df_test$p_multi[df_test$is_tail], df_test$y[df_test$is_tail])

lab_all  <- sprintf("Î”AUC = %.3f  (p=%.3g)", as.numeric(roc_multi_all$auc - roc_mono_all$auc), test_all$p.value)
lab_tail <- sprintf("Î”AUC = %.3f  (p=%.3g)", as.numeric(roc_multi_tail$auc- roc_mono_tail$auc), test_tail$p.value)

theme_pub <- theme_minimal(base_size=12) +
  theme(plot.title=element_text(face="bold", size=13), legend.position="bottom")

# Error vs Density (tail shaded)
err_mono_pt  <- abs((df_test$p_mono >0.5) - df_test$y)
err_multi_pt <- abs((df_test$p_multi>0.5) - df_test$y)
dens <- df_test$density
tail_band <- range(dens[df_test$is_tail])
p_err_dens2 <- ggplot(data.frame(density=dens, Monolith=err_mono_pt, Multi=err_multi_pt))+
  annotate("rect", xmin=tail_band[1], xmax=tail_band[2], ymin=-Inf, ymax=Inf,
           alpha=0.08, fill="grey20")+
  geom_smooth(aes(density, Monolith), se=FALSE) +
  geom_smooth(aes(density, Multi), linetype="dashed", se=FALSE) +
  labs(title="Error vs Local Density",
       subtitle="Shaded = tail region used in evaluation",
       x="Local density (higher = more typical)", y="0/1 error (smoothed)") +
  theme_pub

# Riskâ€“coverage (confidence = 10 for rare w/ x3, else |p_multi-0.5|)
conf_multi <- ifelse(!is.na(df_test$x3), 10, abs(df_test$p_multi - 0.5))
ord <- order(conf_multi, decreasing=TRUE)
cov_seq <- seq(0.4, 1.0, by=0.05)
risk_cov <- lapply(cov_seq, function(cv){
  k <- ceiling(cv*nrow(df_test)); ix <- ord[1:k]
  data.frame(coverage=cv,
             err_mono = mean( (df_test$p_mono [ix]>0.5)!=df_test$y[ix] ),
             err_multi= mean( (df_test$p_multi[ix]>0.5)!=df_test$y[ix] ))
}) %>% dplyr::bind_rows()
gap80 <- with(risk_cov[abs(risk_cov$coverage-0.8)<1e-8,], err_mono - err_multi)

p_risk2 <- ggplot(risk_cov, aes(x=coverage))+
  geom_line(aes(y=err_mono), size=1)+
  geom_line(aes(y=err_multi), linetype="dashed", size=1)+
  annotate("text", x=.8, y=min(risk_cov$err_multi)+0.01,
           label=sprintf("Gap @80%% cov: %.3f", gap80), hjust=0, size=3.5)+
  labs(title="Riskâ€“Coverage", subtitle="solid=monolith, dashed=multi-agent",
       x="Coverage", y="0/1 error among kept")+
  theme_pub

# Tail calibration with ECE
cal_plot <- function(p, y, lab){
  br <- quantile(p, probs=seq(0,1,length.out=17), na.rm=TRUE)
  br[1] <- min(br[1],0); br[length(br)] <- max(br[length(br)],1)
  b <- cut(p, breaks=unique(br), include.lowest=TRUE)
  d <- aggregate(list(pred=p, obs=y), by=list(bin=b), FUN=mean, na.rm=TRUE)
  d$N <- as.numeric(table(b)[as.character(d$bin)])
  d$model <- lab
  d
}
d1 <- cal_plot(df_test$p_mono [df_test$is_tail], df_test$y[df_test$is_tail], "Monolith")
d2 <- cal_plot(df_test$p_multi[df_test$is_tail], df_test$y[df_test$is_tail], "Multi-agent")
dd <- rbind(d1,d2)
p_cal2 <- ggplot(dd, aes(x=pred, y=obs, size=N, shape=model))+
  geom_point(alpha=0.9)+
  geom_abline(slope=1, intercept=0, linetype="dotted")+
  annotate("text", x=0.05, y=0.95,
           label=sprintf("ECE (tail): mono=%.3f, multi=%.3f",
                         ece_mono_tail, ece_multi_tail),
           hjust=0, size=3.6)+
  labs(title="Tail calibration-in-the-small",
       x="Mean predicted prob (bin)", y="Observed outcome rate")+
  theme_pub

# Cluster D diagnostic (show x3 explains win)
dD <- df_test[df_test$cluster=="D", c("x3","p_mono","p_multi","y")]
dD_long <- tidyr::pivot_longer(dD, p_mono:p_multi, names_to="model", values_to="prob")
dD_long$model <- factor(dD_long$model,
                        levels=c("p_mono","p_multi"),
                        labels=c("Monolith (no x3)","Multi-agent specialist (x3)"))
p_x3 <- ggplot(dD_long, aes(x=x3, y=prob, color=model))+
  geom_point(alpha=0.25, size=1)+
  geom_smooth(se=FALSE)+
  labs(title="Rare cluster (D): probability vs x3",
       subtitle="Specialist uses x3; monolith cannot",
       x="x3 (rare-only measurement)", y="Predicted probability")+
  theme_pub + theme(legend.position="bottom")

# --- Rare cluster D: ROC panel (adds to multi-panel) --------------------------
# Recompute if not available already
if(!exists("roc_mono_D") || !exists("roc_multi_D") || !exists("test_D")){
  ixD <- which(df_test$cluster == "D")
  roc_mono_D  <- pROC::roc(df_test$y[ixD], df_test$p_mono[ixD],  quiet = TRUE)
  roc_multi_D <- pROC::roc(df_test$y[ixD], df_test$p_multi[ixD], quiet = TRUE)
  test_D      <- pROC::roc.test(roc_mono_D, roc_multi_D, method = "delong")
}
lab_D <- sprintf("Î”AUC = %.3f  (p=%.3g)",
                 as.numeric(roc_multi_D$auc - roc_mono_D$auc), test_D$p.value)
roc_list_D <- list(`Monolith` = roc_mono_D, `Multi-agent` = roc_multi_D)
p_rocD <- pROC::ggroc(roc_list_D) +
  labs(title = "Rare cluster D: ROC",
       subtitle = lab_D,
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_pub

# Metric bars with Î” and p-values
metrics <- data.frame(
  metric = factor(c("Acc (overall)","Acc (tail)","AUC (overall)","AUC (tail)"),
                  levels=c("Acc (overall)","Acc (tail)","AUC (overall)","AUC (tail)")),
  Monolith = c(acc_mono, acc_mono_tail, as.numeric(roc_mono_all$auc),  as.numeric(roc_mono_tail$auc)),
  Multi    = c(acc_multi, acc_multi_tail, as.numeric(roc_multi_all$auc), as.numeric(roc_multi_tail$auc))
) %>% tidyr::pivot_longer(Monolith:Multi, names_to="model", values_to="value")
p_bars2 <- ggplot(metrics, aes(x=metric, y=value, fill=model))+
  geom_col(position="dodge", width=0.7)+
  geom_text(aes(label=sprintf("%.3f", value)),
            position=position_dodge(width=0.7), vjust=-0.25, size=3)+
  annotate("text", x=3.5, y=max(metrics$value)+0.02, size=3.5,
           label=paste("Overall", lab_all, "\nTail", lab_tail))+
  coord_cartesian(ylim=c(0, 1.05*max(metrics$value)))+
  labs(title="Population vs Tail: averages hide the edges", y="Value", x=NULL)+
  theme_pub

# ---- Assemble & export multi-panel (with D ROC) ------------------------------
panel <- (p_err_dens2 | p_risk2) /
  (p_cal2 | p_x3) /
  (p_rocD | p_bars2) +
  plot_annotation(
    title    = "N-of-1 Simulation: Multi-agent protects rare cases",
    subtitle = "Tail shading, Î”AUC p-values, tail ECE, x3 diagnostic, and cluster-D ROC"
  )

if(!dir.exists("sim_outputs")) dir.create("sim_outputs")
ggplot2::ggsave("sim_outputs/panel_camera_ready.png", panel, width=12, height=12, dpi=300)
ggplot2::ggsave("sim_outputs/panel_camera_ready.pdf", panel, width=12, height=12)

write.csv(results_tbl_fmt, "sim_outputs/results_table.csv", row.names = FALSE)
write.csv(as.data.frame(per_cluster), "sim_outputs/per_cluster_metrics.csv", row.names = FALSE)

cat("\nExported panel and tables to ./sim_outputs/\n")

# ======================= 8) Ablation (what matters?) ==========================
eval_metrics <- function(p, y, tail_idx){
  list(
    AUC_all  = as.numeric(pROC::roc(y, p, quiet=TRUE)$auc),
    ACC_all  = mean((p>0.5)==y),
    AUC_tail = as.numeric(pROC::roc(y[tail_idx], p[tail_idx], quiet=TRUE)$auc),
    ACC_tail = mean((p[tail_idx]>0.5)==y[tail_idx])
  )
}

Wprox     <- ap_test$Wprox
p_agentsW <- rowSums(Wprox * P_test)    # proximity-weighted agents
p_agentsE <- rowMeans(P_test)           # equal-weighted agents

p_mono    <- pred_mono_test
p_full    <- p_multi
p_stack   <- predict(meta_fit,
                     newdata=cbind(test[,c("density","disagree")], as.data.frame(P_test)),
                     type="response")

p_no_specialist <- p_stack
p_no_stack      <- ifelse(!is.na(test$x3),
                          predict(rare_model, newdata=test, type="response"),
                          p_agentsW)
p_agents_only   <- p_agentsE

abl <- list(
  Monolith         = eval_metrics(p_mono,          test$y, tail_ix),
  Multi_full       = eval_metrics(p_full,          test$y, tail_ix),
  No_specialist    = eval_metrics(p_no_specialist, test$y, tail_ix),
  No_stacking      = eval_metrics(p_no_stack,      test$y, tail_ix),
  Agents_only      = eval_metrics(p_agents_only,   test$y, tail_ix)
)
abl_tbl <- do.call(rbind, lapply(names(abl), function(k) cbind(Model=k, as.data.frame(abl[[k]]))))
abl_tbl <- dplyr::mutate(abl_tbl, dplyr::across(where(is.numeric), ~round(., 3)))
cat("\n==== Ablation table (same TEST) ====\n"); print(abl_tbl, row.names=FALSE)
write.csv(abl_tbl, "sim_outputs/ablation_metrics.csv", row.names=FALSE)

# ======================= 9) Paired bootstrap on TEST =========================
set.seed(999)
B <- 1000
boot_one <- function(idx_vec){
  yb <- test$y[idx_vec]
  pm <- p_mono[idx_vec]
  pf <- p_full[idx_vec]
  
  S_b  <- cov(test[idx_vec, c("x1","x2")]); mu_b <- colMeans(test[idx_vec, c("x1","x2")])
  md_b <- mahalanobis(test[idx_vec, c("x1","x2")], center=mu_b, cov=S_b)
  tail_b <- order(md_b, decreasing=TRUE)[1:ceiling(0.12*length(idx_vec))]
  
  auc_all_delta  <- as.numeric(pROC::roc(yb, pf, quiet=TRUE)$auc -
                                 pROC::roc(yb, pm, quiet=TRUE)$auc)
  acc_all_delta  <- mean((pf>0.5)==yb) - mean((pm>0.5)==yb)
  auc_tail_delta <- as.numeric(pROC::roc(yb[tail_b], pf[tail_b], quiet=TRUE)$auc -
                                 pROC::roc(yb[tail_b], pm[tail_b], quiet=TRUE)$auc)
  acc_tail_delta <- mean((pf[tail_b]>0.5)==yb[tail_b]) -
    mean((pm[tail_b]>0.5)==yb[tail_b])
  
  c(auc_all_delta, acc_all_delta, auc_tail_delta, acc_tail_delta)
}

n <- nrow(test)
idx_mat  <- replicate(B, sample.int(n, n, replace=TRUE))
boot_mat <- t(apply(idx_mat, 2, boot_one))
colnames(boot_mat) <- c("dAUC_all","dACC_all","dAUC_tail","dACC_tail")

ci <- function(v) quantile(v, c(0.025,0.5,0.975))
cis <- apply(boot_mat, 2, ci)
pvals <- apply(boot_mat, 2, function(v) 2*min(mean(v<=0), mean(v>=0)))  # two-sided

boot_tbl <- data.frame(
  Metric = c("Î”AUC overall","Î”ACC overall","Î”AUC tail","Î”ACC tail"),
  `2.5%` = round(cis[1,], 3),
  Median = round(cis[2,], 3),
  `97.5%`= round(cis[3,], 3),
  `p (paired bootstrap)` = signif(pvals, 3)
)
cat("\n==== Paired bootstrap (B=1000) on TEST ====\n"); print(boot_tbl, row.names=FALSE)
write.csv(boot_tbl, "sim_outputs/bootstrap_deltas.csv", row.names=FALSE)

cat("\nAll artifacts saved in ./sim_outputs/  âœ…\n")
