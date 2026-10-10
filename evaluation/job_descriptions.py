"""One short job description per category in the Kaggle "Resume Dataset" (UpdatedResumeDataSet.csv).

Written to read like real postings, not keyword lists, so the evaluation
tests whether a method understands a job, not whether it matches a label.
"""

JOB_DESCRIPTIONS = {
    "Data Science": "We need a data scientist to build predictive models, run experiments and explain results to the business. You will clean large datasets, engineer features and train machine learning models in Python, and communicate findings with clear visualisations.",
    "HR": "Join our people team to run end-to-end hiring, onboarding and employee engagement. You will screen candidates, coordinate interviews, maintain employee records, handle payroll queries and support performance reviews.",
    "Advocate": "Our firm is looking for an advocate to represent clients in court, draft petitions, agreements and legal notices, conduct legal research and advise clients on civil and criminal matters.",
    "Arts": "We are seeking a creative professional with a background in fine arts, drawing and painting to lead workshops, curate exhibitions and produce original artwork and illustrations.",
    "Web Designing": "Looking for a web designer to create attractive, responsive websites. You will design page layouts and user interfaces, build pages with HTML, CSS and JavaScript, and work with Photoshop for graphics.",
    "Mechanical Engineer": "We are hiring a mechanical engineer to design components, prepare 2D and 3D drawings, oversee manufacturing and maintenance of machines, and improve production quality on the shop floor.",
    "Sales": "We want a sales executive to find new customers, manage key accounts, meet monthly revenue targets, negotiate deals and build long-term client relationships in the field.",
    "Health and fitness": "Seeking a fitness trainer to design personalised workout and nutrition plans, coach clients one-on-one and in groups, track their progress and promote healthy lifestyle habits.",
    "Civil Engineer": "We need a civil engineer to plan and supervise construction sites, prepare estimates and quantity surveys, check structural drawings and ensure buildings and roads are built to specification.",
    "Java Developer": "Hiring a Java developer to build backend services with Spring Boot and Hibernate, write REST APIs, work with relational databases and maintain enterprise web applications.",
    "Business Analyst": "We are looking for a business analyst to gather requirements from stakeholders, document business processes, write functional specifications and work with development teams to deliver solutions.",
    "SAP Developer": "We need an SAP developer to build and enhance reports, interfaces and forms in SAP using ABAP, support SAP modules and work with functional consultants on implementations.",
    "Automation Testing": "Looking for a test automation engineer to design automated test suites with Selenium, build test frameworks, run regression tests in CI pipelines and report defects.",
    "Electrical Engineering": "We are hiring an electrical engineer to design and maintain electrical systems, panels and power distribution, handle installation and commissioning, and ensure safety standards on site.",
    "Operations Manager": "Seeking an operations manager to run day-to-day operations, manage teams and vendors, improve processes, control costs and make sure service levels and targets are met.",
    "Python Developer": "We need a Python developer to build web applications and APIs with Django or Flask, write clean, tested code, work with databases and automate tasks with scripts.",
    "DevOps Engineer": "Hiring a DevOps engineer to automate build and deployment pipelines, manage cloud infrastructure, containerise applications with Docker and Kubernetes, and monitor production systems.",
    "Network Security Engineer": "We are looking for a network security engineer to configure firewalls, routers and switches, monitor networks for threats, manage VPNs and respond to security incidents.",
    "PMO": "Our project management office needs a professional to track project plans, budgets and risks, prepare status reports for leadership, and coordinate governance across multiple projects.",
    "Database": "Hiring a database administrator to install, tune and back up databases, write complex SQL and stored procedures, manage user access and keep production databases fast and reliable.",
    "Hadoop": "We need a big data engineer with Hadoop experience to build data pipelines on HDFS, process large datasets with Hive, Pig and Spark, and manage the cluster.",
    "ETL Developer": "Looking for an ETL developer to design data integration workflows that extract, transform and load data into a data warehouse, using ETL tools such as Informatica and writing SQL.",
    "DotNet Developer": "We are hiring a .NET developer to build web applications with C#, ASP.NET MVC and SQL Server, and maintain existing enterprise software.",
    "Blockchain": "Seeking a blockchain developer to write and test smart contracts, build decentralised applications on Ethereum and integrate them with web front ends.",
    "Testing": "We need a software tester to write test cases, carry out manual functional and regression testing, log and track bugs, and make sure releases meet quality standards.",
}
