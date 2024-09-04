from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Open the text file and read its contents
with open('text_file.txt', 'r') as file:
    text = file.read()

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=800, background_color='white', max_words=50, contour_width=3, contour_color='steelblue')

# Generate the word cloud
wordcloud.generate(text)

# Visualize the word cloud
plt.figure(figsize=(8,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

# Show the plot
plt.show()
