#Business Intelligence Analytics Portfolio
Jupyter , Python , SQL , Excel,  Tableau

### What has the prject achieved
 It takes vague business problem, investigates them independently, and delivers findings that drive decisions
 

What This Project Is
With 35,000+ e-commerce orders spanning 6 regions, 27 marketing channels, and 6 years of data.
    Different stakeholder questions from board and technical teams. 
    Two full investigations. 
    Dashboards. 
    Presentations 


### Investigate
1. Why Are We Losing Revenue to Refund? [Tableau Visualisation Dashboard and charts ](https://public.tableau.com/views/RefundDashboard_17714614553130/Dashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
    The process
    
        - Cleaned and standardised 5 different date formats across the same columns using Python
        - Converted multi-currency prices to USD using live exchange rates via yfinance API
        - Formed and tested the hypothesis that slow delivery was driving refunds
        - Killed that hypothesis when the data contradicted it — fast delivery regions were refunding more
        - Identified the real problem: a product expectation gap in mature markets (North America, Europe)
        - Quantified $264,719 in lost revenue and traced it to specific product categories and channels
        - Built a Tableau dashboard with filter actions for live stakeholder Q&A
        - Delivered a 5-minute boardroom presentation and a technical video walkthrough
    
    ##### Finding
   Electronics customers acquired through Organic, Direct and Paid Search in North America have the highest refund rate not
   because delivery is slow, but because the product doesn't match the expectation of a high-intent buyer.
   That is a listings problem, not a logistics problem.

2.  Which Customers Are Worth Acquiring?[Customer Acquisition Dashboard and charts ](https://public.tableau.com/views/CustomerAcquisitionVisualisationchartsanddashboard/CustomerDashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
The brief: "We're spending money across multiple channels. Which customers actually bring value?"
Process:

        - Segmented 35,000+ customers by acquisition method, purchase platform, region, and behavior simultaneously
        - Built a quadrant scatter plot (Avg Order Value vs Refund Rate) to classify 27 channels into invest / watch / cut buckets
        - Built a region × platform heatmap to surface the worst-performing intersections
        - Identified that Email signup customers generate 3× more revenue than any other segment — but also carry the highest refund risk
        - Found Mobile App has a refund rate 6× lower than Desktop across every region globally
        - Identified Middle East Mobile Web at 50% refund rate — flagged as a broken experience requiring immediate investigation
        - Delivered actionable segment profiles to marketing and product teams

    ##### Finding
    The best customers are already in the business.
    The problem is we are landing them on the wrong platform and not protecting them with accurate product information.

Skills Demonstrated
| Skill | Where |
| --- | --- |
| Python — Pandas, dateutil, yfinance | Data cleaning, currency conversion, EDA |
| Data wrangling — messy real-world formats | 5 date formats normalised, nulls handled, currency extracted |
| Hypothesis-driven analysis | Two hypotheses formed, tested, one killed and revised by data |
| Tableau — calculated fields, dashboard actions | Two dashboards built from scratch with narrative flow |
| Stakeholder communication | Scripts written for CFO-level and technical audiences separately |
| Business framing | Translated vague briefs into specific, actionable findings |
| Anomaly detection | 50% refund rate intersection surfaced and flagged |
| Customer segmentation | Cohorts profiled by acquisition method, platform, region, behavior |





The Role fit for :
Business Intelligence Analyst — dashboards, KPIs, stakeholder reporting, revenue tracking
Product Analyst — user behavior segmentation, platform performance, feature adoption thinking
Commercial and Growth Analyst — channel quality, customer acquisition ROI, retention signals
Category Analyst — product-level performance, refund analysis, margin impact
The core skill across all four is the same: take an ambiguous business question, investigate it with data, and come back with something the business can act on.

Tools
Python Pandas Tableau Excel Jupyter yfinance dateutil

Links :
Tableau 
    [Tableau Visualisation Dashboard and charts ](https://public.tableau.com/views/RefundDashboard_17714614553130/Dashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
    [Customer Acquisition Dashboard and charts ](https://public.tableau.com/views/CustomerAcquisitionVisualisationchartsanddashboard/CustomerDashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
    
Notebook [Collab]
    [Notebook](https://colab.research.google.com/drive/13c1qyKD9KVETt0AhqBlt5HSjSjmzB-uN?usp=sharing)
Youtube Presentations 
    [Refund Rate Link](https://youtu.be/i5Bif0ocDdw)

Folder 
Data - both clean and original data (note: the cleaner version has been manipulated expeditiously using pandas python and excel a recording would mixed up everything therefore chose to leave it aside and post tableau visualisation and dashboard presentation)

Detailed methodology and findings are documented in each subfolder README.
