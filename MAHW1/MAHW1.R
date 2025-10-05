library(ggplot2)
library(ggpubr)

# --- Load data ---------------------------------------------------------
df <- read.csv("/Users/serinepoghosyan/Downloads/action_cam_shipments.csv", stringsAsFactors = FALSE)

# Ensure ascending years and contiguous time index t = 1..T
df <- df[order(df$year), ]
df$t   <- seq_len(nrow(df))          # 1,2,...,T
sales  <- df$sales                   # a(t): million units per year
t      <- df$t

# --- Bass formulas (from slides) --------------------------------------
bass.f <- function(t, p, q){
  # fraction adopting at time t
  ((p+q)^2 / p) * exp(-(p+q)*t) / (1 + (q/p) * exp(-(p+q)*t))^2
}
bass.F <- function(t, p, q){
  # cumulative fraction by time t
  (1 - exp(-(p+q)*t)) / (1 + (q/p) * exp(-(p+q)*t))
}
t_star <- function(p, q){
  # peak time (continuous)
  log(q/p) / (p + q)
}

# --- NLS estimation: sales ~ M * f(t; p, q) ---------------------------
start_M <- sum(sales)              # initial guesses
start_p <- 0.02
start_q <- 0.40

nls_fit <- nls(
  sales ~ M * (((p+q)^2 / p) * exp(-(p+q)*t) / (1 + (q/p) * exp(-(p+q)*t))^2),
  start = list(M = start_M, p = start_p, q = start_q),
  control = list(maxiter = 2000, warnOnly = TRUE)
)

summary(nls_fit)
coef_nls <- coef(nls_fit)
M_hat <- as.numeric(coef_nls["M"])
p_hat <- as.numeric(coef_nls["p"])
q_hat <- as.numeric(coef_nls["q"])

cat(sprintf("\nEstimated (NLS):  M = %.3f,  p = %.5f,  q = %.5f\n",
            M_hat, p_hat, q_hat))
cat(sprintf("Peak time t* (continuous) = %.2f periods\n", t_star(p_hat, q_hat)))


# Fitted values and residuals
sales_hat <- M_hat * bass.f(t, p_hat, q_hat)
resid     <- sales - sales_hat

#Observed vs fitted (per year)
library(ggplot2)
p_fit <- ggplot(df, aes(x = year, y = sales)) +
  geom_col(fill = "grey70") +
  geom_line(aes(y = sales_hat), linewidth = 1.1, color = "black") +
  labs(title = "Observed vs Bass fitted — digital camera shipments",
       x = "Year", y = "Million units") +
  theme_minimal()
ggsave("img/cipa_bass_fit.png", p_fit, width = 8, height = 4.5, dpi = 150)

#Cumulative observed vs Bass implied
cum_obs <- cumsum(sales)
cum_hat <- M_hat * bass.F(t, p_hat, q_hat)
p_cum <- ggplot(data.frame(year=df$year, cum_obs, cum_hat), aes(x=year)) +
  geom_line(aes(y = cum_obs)) +
  geom_line(aes(y = cum_hat), linetype = 2) +
  labs(title = "Cumulative: observed (solid) vs Bass implied (dashed)",
       x = "Year", y = "Million units (cumulative)") +
  theme_minimal()
ggsave("img/cipa_bass_cumulative.png", p_cum, width = 8, height = 4.5, dpi = 150)



# ---- Inputs you choose ------------------------------------------------
launch_year <- 2025             # set start year for the new product's diffusion
horizon     <- 10               # forecast length in years

# Market potential (in million units). 
M_base <- 20                   
M_low  <- M_base * 0.8
M_high <- M_base * 1.2

# ---- Build a fresh time axis for the NEW innovation -------------------
t_new    <- 1:horizon
years_fc <- launch_year + (t_new - 1)

# ---- Using the look-alike's p,q we already estimated -------------------
# (From previous fit)
# p_hat <- 0.0050
# q_hat <- 0.3855

# Annual and cumulative forecasts for three M scenarios
a_base  <- M_base * bass.f(t_new, p_hat, q_hat)
A_base  <- M_base * bass.F(t_new, p_hat, q_hat)

a_low   <- M_low  * bass.f(t_new, p_hat, q_hat)
A_low   <- M_low  * bass.F(t_new, p_hat, q_hat)

a_high  <- M_high * bass.f(t_new, p_hat, q_hat)
A_high  <- M_high * bass.F(t_new, p_hat, q_hat)

# Peak (in NEW product time scale)
t_star_new <- t_star(p_hat, q_hat)            
peak_year_new <- launch_year + t_star_new - 1

# ---- Output tables ----------------------------------------------------
forecast_tbl <- data.frame(
  year      = years_fc,
  adopters_base_m = a_base,
  adopters_low_m  = a_low,
  adopters_high_m = a_high,
  cum_base_m      = A_base,
  cum_low_m       = A_low,
  cum_high_m      = A_high
)

dir.create("data", showWarnings = FALSE)
write.csv(forecast_tbl, "data/innovation_forecast.csv", row.names = FALSE)
print(forecast_tbl, row.names = FALSE)

cat(sprintf("\nPeak time (t*) ≈ %.2f years after launch → approx peak year ≈ %.1f\n",
            t_star_new, peak_year_new))

# ----- Inputs ----------------------------------------------------------
launch_year <- 2025     # first adoption year for the innovation
horizon     <- 10       # how many years to list 

# Market potential in million units (global) ±20% sensitivity.
M_base <- 20
M_low  <- M_base * 0.8
M_high <- M_base * 1.2

# ----- Time axis for the NEW product ----------------------------------
t_new    <- 1:horizon
years_fc <- launch_year + (t_new - 1)

# ----- Annual & cumulative (base / low / high) ------------------------
a_base <- M_base * bass.f(t_new, p_hat, q_hat)
A_base <- M_base * bass.F(t_new, p_hat, q_hat)

a_low  <- M_low  * bass.f(t_new, p_hat, q_hat)
A_low  <- M_low  * bass.F(t_new, p_hat, q_hat)

a_high <- M_high * bass.f(t_new, p_hat, q_hat)
A_high <- M_high * bass.F(t_new, p_hat, q_hat)

# Peak timing (years after launch) and approximate calendar peak year
t_star_new    <- t_star(p_hat, q_hat)
peak_year_new <- launch_year + t_star_new - 1

# ----- Assembling a table --------------------------------------------
adopters_tbl <- data.frame(
  year           = years_fc,
  adopters_low_m = a_low,
  adopters_base_m= a_base,
  adopters_high_m= a_high,
  cum_low_m      = A_low,
  cum_base_m     = A_base,
  cum_high_m     = A_high
)


adopters_tbl_out <- within(adopters_tbl, {
  adopters_low_m  <- round(adopters_low_m, 3)
  adopters_base_m <- round(adopters_base_m, 3)
  adopters_high_m <- round(adopters_high_m, 3)
  cum_low_m       <- round(cum_low_m, 3)
  cum_base_m      <- round(cum_base_m, 3)
  cum_high_m      <- round(cum_high_m, 3)
})

# Save 
dir.create("data", showWarnings = FALSE)
write.csv(adopters_tbl_out, "data/innovation_adopters_by_year.csv", row.names = FALSE)


print(adopters_tbl_out, row.names = FALSE)
cat(sprintf("\nPeak ~ %.2f years after launch → approx peak calendar year ~ %.1f\n",
            t_star_new, peak_year_new))



