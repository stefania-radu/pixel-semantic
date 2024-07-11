import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the style and font size
sns.set_style("darkgrid")


data = {
    "Language": ["Amharic", "Arabic", "Bengali", "English", "Finnish", "Hausa", "Igbo", "Indonesian", "Italian",
                 "Kinyarwanda", "Korean", "Luganda", "Naija Pidgin", "Norwegian", "Romanian", "Russian", "Swahili",
                 "Telugu", "Wolof", "Yorùbá"],
    "ISO 639-3": ["AMH", "ARA", "BEN", "ENG", "FIN", "HAU", "IBO", "IND", "ITA", "KIN", "KOR", "LUG", "PCM", "NOR",
                  "RON", "RUS", "SWA", "TEL", "WOL", "YOR"],
    "Language Family": ["Afro-Asiatic", "Afro-Asiatic", "Indo-European", "Indo-European", "Uralic", "Afro-Asiatic",
                        "Niger-Congo", "Austronesian", "Indo-European", "Niger-Congo", "Koreanic", "Niger-Congo",
                        "English Creole", "Indo-European", "Indo-European", "Indo-European", "Niger-Congo",
                        "Dravidian", "Niger-Congo", "Niger-Congo"],
    "Script": ["Ge'ez", "Arabic", "Bengali", "Latin", "Latin", "Latin", "Latin", "Latin", "Latin", "Latin", "Korean",
               "Latin", "Latin", "Latin", "Latin", "Cyrillic", "Latin", "Telugu", "Latin", "Latin"],
    "Pre-training": [True, True, True, True, True, True, True, True, False, True, True, True, True, False, False, True,
                     True, True, True, True],
    "Fine-tuning": [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                    True, True, True, True]
}

# Create DataFrame
df = pd.DataFrame(data)

# Prepare data for stacked histogram
df['Both'] = df['Pre-training'] & df['Fine-tuning']
df['Only Fine-tuning'] = ~df['Pre-training'] & df['Fine-tuning']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

colors = ['#ffc200', '#ff8200']  # Yellow and Orange

# Subplot 1: Distribution of languages per script
df.groupby('Script')[['Pre-training', 'Only Fine-tuning']].sum().plot(kind='barh', stacked=True, ax=ax1, color=colors)
ax1.set_xlabel('Count', fontsize=16)
ax1.set_ylabel('Script', fontsize=16)
ax1.legend(['Pre-training', 'Fine-tuning'], title='Category', fontsize=14)

# Subplot 2: Distribution of languages per language family
df.groupby('Language Family')[['Pre-training', 'Only Fine-tuning']].sum().plot(kind='barh', stacked=True, ax=ax2, color=colors)
ax2.set_xlabel('Count', fontsize=16)
ax2.set_ylabel('Language Family', fontsize=16)
ax2.legend(['Pre-training', 'Fine-tuning'], title='Category', fontsize=14)

plt.tight_layout()

plt.savefig("languages_analysis.pdf", bbox_inches='tight')
plt.close()
