# Bass Diffusion Model Analysis — Insta360 X4 (2024)

### Course: Marketing Analytics  
### Author: Serine Poghosyan  
### Date: 05/10/2025  

---

## 🧠 Project Overview
This project applies the **Bass Diffusion Model** to forecast the global market adoption of the **Insta360 X4**, one of *TIME’s Best Inventions of 2024*.  
The study uses the **GoPro MAX (2019)** as the historical analogue to estimate the diffusion parameters (p and q) and leverages **CIPA digital camera shipment data (1999–2024)** for fitting and validation. The model captures the rise, peak, and decline of the digital camera market and extends those dynamics to predict adoption for the new 360° action camera generation.

---

## 📊 Methodology Summary

| Step | Description |
|------|--------------|
| **1. Choose an Innovation** | Selected the Insta360 X4 (2024) — a compact 360° action camera for creators, athletes, and travelers. |
| **2. Select an Analogue** | Used the GoPro MAX (2019) as the historical analogue, which shares core 360° capture and stabilization technology. |
| **3. Gather Data** | Annual global digital camera shipment data from the **Camera & Imaging Products Association (CIPA)**, 1999–2024. |
| **4. Estimate Bass Model** | Fitted Bass model parameters (p, q, M) to CIPA data using nonlinear least squares in R. |
| **5. Forecast Diffusion** | Simulated 10-year adoption (2025–2034) of the Insta360 X4 using transferred Bass parameters. |
| **6. Scope** | Global market scope — consistent with worldwide CIPA data and Insta360’s international distribution. |
| **7. Estimate Adopters by Period** | Computed annual and cumulative adopters using Bass equations under base, low, and high (±20%) market-potential scenarios. |

---

## ⚙️ Key Parameters and Insights

| Parameter | Symbol | Estimate |
|------------|---------|-----------|
| Coefficient of innovation | p | 0.0050 |
| Coefficient of imitation | q | 0.3855 |
| Market potential (historical analog) | M | 1,214 million units |
| Predicted market potential (Insta360 X4) | M | 20 million units |
| Peak adoption time | t* | ≈ 11.1 years after launch |

**Interpretation:**  
The low *p* and high *q* indicate that adoption of digital imaging devices was primarily driven by **imitation and word-of-mouth** rather than advertising. The diffusion curve matches the observed digital camera peak (~2010) and decline thereafter.

For the **Insta360 X4**, diffusion is expected to peak around **2035**, with approximately **1.9M units sold annually** and cumulative adopters reaching **~7.7M (≈38% of M)** within 10 years of launch.

---

## 📈 Visualization Outputs

### 1️⃣ Observed vs. Bass Fitted (Digital Cameras)
![Bass Fit – CIPA](cipa_bass_fit.jpeg)

### 2️⃣ Cumulative Observed vs. Bass Implied
![Cumulative Bass Fit](cipa_bass_cumulative.jpeg)

These figures demonstrate that the Bass model accurately replicates the lifecycle of digital camera shipments, validating parameter use for new-product forecasting.

---

## 🧩 Repository Structure
```
MAHW1/
│
├── data/
│   ├── action_cam_shipments.csv               # Processed digital camera data (CIPA 1999–2024)
│   ├── innovation_adopters_by_year.csv        # Bass forecast for Insta360 X4 (annual + cumulative)
│   └── innovation_forecast.csv                # 10-year forecast under base/low/high M scenarios    
│
├── img/
│   ├── cipa_bass_fit.jpeg                     # Observed vs Bass fitted plot
│   └── cipa_bass_cumulative.jpeg              # Cumulative adoption comparison               
│
├── report/
│   ├── MAHW1.Rmd                              # Full R Markdown report with analysis & commentary
│   └── MAHW1.pdf                              # Final report (knitted output)   
│
├── MAHW1.R                                # Main R script (parameter estimation + forecast)              
└── README.md                              # Project summary and reproduction guide
```


---

## 🧮 How to Reproduce

1. **Install R packages**
   ```r
   install.packages(c("ggplot2", "dplyr", "readr", "minpack.lm"))
   ```

2. **Run the analysis script**
   ```r
   source("MAHW1.R")
   ```

3. **Alternatively, knit the report**
   ```r
   rmarkdown::render("MAHW1.Rmd")
   ```

This reproduces all figures, parameter estimates, and forecast tables.

---

## 📚 References

- **TIME (2024).** *Insta360 X4.*  
  [https://time.com/collection/best-inventions-2024/](https://time.com/collection/best-inventions-2024/)
- **Camera & Imaging Products Association (CIPA).** *Production and Shipment of Digital Still Cameras (1999–2024).*  
  [https://www.cipa.jp/e/stats/dc-2024.html](https://www.cipa.jp/e/stats/dc-2024.html)
- **Bass, F. M. (1969).** *A New Product Growth Model for Consumer Durables.* *Management Science, 15*(5), 215–227.
- R Libraries: *ggplot2, dplyr, readr, minpack.lm, rmarkdown.*

---

## 🏁 Summary
This project models the diffusion of **Insta360 X4 (2024)** using historical digital camera shipment data and Bass model estimation.  
By leveraging real-world data and validated diffusion parameters, it provides a realistic, quantitative forecast of the 360° camera market’s growth trajectory through 2035.
