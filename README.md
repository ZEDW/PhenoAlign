# PhenoAlign
PhenoAlign is a tool designed for aligning medical text phenotype information. We integrate the pre-trained language model BERT into a knowledge-based approach. The execution of PhenoAlign entails the following two steps:

1.Download the pre-trained model SimClassifier from https://huggingface.co/YangTaoM/SimClassifier#/.

2.Place the SimClassifier model within the PhenoAlign directory and execute the ../model/reFunctions_align_all.py code.

# Annotation guideline：

The flowchart for aligning phenotype information is shown in Annotation Guideline.svg.

PhenoSSU is an information template for characterizing phenotypic details that contains 17 attributes related to phenotypic concepts. These attributes can be divided into two categories according to the phenotypic details they characterized: (1) the attributes of the phrase-type phenotype, such as "severe cough" or "fever", including assertion, severity, frequency in population, temporary pattern, age specificity, sex specificity, characteristic of pain, time of duration, aggregating factors, releasing factors, body location, spatial pattern, polarity, and clinical stage for disease, and (2) the characteristics of the logic-type phenotype, such as "leukocyte 12.5 × 109/L", including specimen, analyte, and abnormality.

PhenoSSU alignment can be annotated in medical text using the BRAT, a process comprising two consecutive steps: PhenoSSU annotation and PhenoSSU association. Specifically, annotators should initially annotate the complete PhenoSSUs in two medical texts and then associate the "completely identical" and "partially similar" PhenoSSUs according to the semantic relationships of the PhenoSSUs. To illustrate the PhenoSSU alignment annotation, a specific example is provided below.

In this example, for the phrase-type phenotype category, annotators initially annotated the phenotype concept "abdominal pain" within the PhenoSSU. Subsequently, they read through the context of the phenotype to identify the presence of predefined attribute (such as "acute" and "severe") trigger terms and fill the attribute slots with appropriate values indicated by the identified triggers. For the logic-type phenotype category, annotators first annotate the phenotype concept "white blood cell count," then determine abnormalities corresponding to numerical values based on knowledge from laboratory reference ranges (e.g., normal values for white blood cell count are "4-10×10^9/L"), and then fill the attribute slots with appropriate values indicated by the identified triggers.

Following the complete annotation of PhenoSSUs, the semantic relationships between PhenoSSUs in different medical texts are then associated. The semantic relationships between phrase-type PhenoSSUs can be categorized into three types: "completely identical," "partially similar," and "dissimilar." "Completely identical" indicates that two PhenoSSUs are identical at both the core word and attribute word levels. “Partially similar” indicates complete identity at the core word level when the attribute words are not entirely the same or when a hierarchical relationship exists between the core words. "Dissimilar" indicates complete dissimilarity at the core word level. There are two types of semantic relationships between logic-type PhenoSSUs: "completely identical" and "dissimilar." "Completely identical" semantic interpretations identical to those of phrase-type PhenoSSUs, while all other cases are considered "dissimilar."
