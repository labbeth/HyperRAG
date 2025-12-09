'''Defining prompt to detect span related to phenotype in a sentence'''

from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate


# Global template
full_template = """{persona}
{instruction}

{example}

{start}"""
full_prompt = PromptTemplate.from_template(full_template)

# Persona template
persona_template = """You are an experimented {person} with an exhaustive knowledge of human phenotypes ontology."""
persona_prompt = PromptTemplate.from_template(persona_template)

# Instruction template
instruction_template = """Given {text}, you must identify the spans related to possible phenotypes, either explicitly or implicitly.
You should keep in the span all words related to the phenotype that should be informative (such as negation or adjective).
You may reformulate the span if needed.
If you don't detect any span or if you don't know, don't try to make up an answer, just write 'None'."""
instruction_prompt = PromptTemplate.from_template(instruction_template)

# Examples template
example_template = """
SENTENCE: Shortly after birth, he developed tachypnea, irritability and spastic movements of the upper limbs, and he was found to have mild hypocalcemia and hypomagnesemia.
==========
Span: tachypnea, irritability, spastic movements of the upper limbs, mild hypocalcemia, hypomagnesemia

SENTENCE: MR spectroscopy showed a region of increased myoinositol in the left thalamus indicating gliosis with no lactate peak. His TSH has been persistently mildly elevated; however, he is not on thyroxine.
==========
Span: increased myoinositol in the left thalamus, gliosis, TSH has been persistently mildly elevated

SENTENCE: On examination, she has a nonexpressive face with subtle dysmorphism and mild positional deformity of the chest wall. His physical examination was significant for hypertelorism, long thin and hyperextensible fingers and hypotonia. Other examinations were within normal limits. Brain MRI showed diffuse white matter T2 hyperintensity.
==========
Span: nonexpressive face, subtle dysmorphism, mild positional deformity of the chest wall, hypertelorism, long fingers, thin fingers, hyperextensible fingers, hypotonia, diffuse white matter T2 hyperintensity

SENTENCE: He has developed its first seizures at the age of 9 month and continues to seize daily.
==========
Span: seizures, seize daily

SENTENCE: She can only regard faces and smile.
==========
Span: None

SENTENCE: She barely can feel pain, has no tears when she cries even though she has normal sweating.
==========
Span: barely can feel pain, no tears, normal sweating

SENTENCE: She has subtle dysmorphia characterized as hypotelorism and tapering of fingers.
==========
Span: subtle dysmorphia, hypotelorism, tapering of fingers

SENTENCE: Lysosomal enzymes in cultured skin fibroblasts such as beta-galactosidase or total beta-hexosaminidase were within normal limits.
==========
Span: None

SENTENCE: Skeletal survey showed 11 pairs of ribs.
==========
Span: 11 pairs of ribs

SENTENCE: It showed atrophied thalami and restricted water diffusion.
==========
Span: atrophied thalami, abnormal water regulation
"""
example_prompt = PromptTemplate.from_template(example_template)

# Start template
start_template = """SENTENCE: {input}
==========
Span: """
start_prompt = PromptTemplate.from_template(start_template)

# Defining pipeline
input_prompts = [
    ("persona", persona_prompt),
    ("instruction", instruction_prompt),
    ("example", example_prompt),
    ("start", start_prompt)
]
pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)