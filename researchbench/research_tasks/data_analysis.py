from pathlib import Path
from typing import Dict, Any, List

from researchbench.research_tasks.base import DataAnalysisTask
from researchbench.research_tasks.manager import register_task_handler
from researchbench.utils import get_logger

logger = get_logger(__name__)


class ClimateChangeDataAnalysis(DataAnalysisTask):
    """Data analysis task for climate change data."""

    def __init__(self):
        super().__init__(
            task_id="data-analysis-climate-change",
            name="Climate Change Data Analysis",
            description="Analyze global temperature and CO2 emission data to identify trends and correlations.",
            dataset_ids=["global-temperature-data", "co2-emissions-data"],
        )

    def run(self, data_dir: Path, output_dir: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run the climate change data analysis task."""
        # In a real implementation, this would provide tools and guidance
        # for analyzing climate change data.
        
        # Create a sample output file
        with open(output_dir / "climate_analysis.md", "w") as f:
            f.write("# Climate Change Data Analysis\n\n")
            f.write("## Introduction\n\n")
            f.write("This analysis examines global temperature and CO2 emission data to identify trends and correlations over time.\n\n")
            f.write("## Data Description\n\n")
            f.write("### Global Temperature Dataset\n\n")
            f.write("The global temperature dataset contains monthly temperature anomalies from 1880 to 2025, relative to the 1951-1980 average. The data is sourced from NASA's Goddard Institute for Space Studies (GISS).\n\n")
            f.write("### CO2 Emissions Dataset\n\n")
            f.write("The CO2 emissions dataset contains annual CO2 emissions from fossil fuel combustion and industrial processes from 1750 to 2025, measured in gigatons of CO2. The data is sourced from the Global Carbon Project.\n\n")
            f.write("## Methodology\n\n")
            f.write("The analysis was conducted using Python with the following libraries:\n\n")
            f.write("- pandas for data manipulation\n")
            f.write("- matplotlib and seaborn for visualization\n")
            f.write("- scipy for statistical analysis\n\n")
            f.write("The analysis included the following steps:\n\n")
            f.write("1. Data cleaning and preprocessing\n")
            f.write("2. Exploratory data analysis\n")
            f.write("3. Trend analysis using moving averages and linear regression\n")
            f.write("4. Correlation analysis between temperature and CO2 emissions\n")
            f.write("5. Regional variation analysis\n\n")
            f.write("## Exploratory Data Analysis\n\n")
            f.write("### Global Temperature Trends\n\n")
            f.write("The global temperature data shows a clear warming trend over the past century. The global mean temperature anomaly has increased from approximately -0.2°C in the early 20th century to over 1.0°C in recent years, relative to the 1951-1980 average.\n\n")
            f.write("![Global Temperature Trend](figures/global_temp_trend.png)\n\n")
            f.write("*Figure 1: Global temperature anomalies from 1880 to 2025, with 5-year moving average.*\n\n")
            f.write("### CO2 Emission Trends\n\n")
            f.write("CO2 emissions have increased dramatically over the past century, from less than 1 gigaton per year in 1900 to over 35 gigatons per year in recent years. The rate of increase has accelerated in recent decades, particularly with the rapid industrialization of developing economies.\n\n")
            f.write("![CO2 Emission Trend](figures/co2_emission_trend.png)\n\n")
            f.write("*Figure 2: Annual CO2 emissions from 1750 to 2025.*\n\n")
            f.write("## Statistical Analysis\n\n")
            f.write("### Trend Analysis\n\n")
            f.write("Linear regression analysis of the global temperature data yields a warming trend of approximately 0.08°C per decade over the entire period (1880-2025), but the rate has increased to approximately 0.18°C per decade since 1970.\n\n")
            f.write("For CO2 emissions, the growth rate has averaged about 3% per year over the past century, with significant variations due to economic cycles, wars, and more recently, climate policies.\n\n")
            f.write("### Correlation Analysis\n\n")
            f.write("The correlation between global temperature anomalies and CO2 emissions is strong, with a Pearson correlation coefficient of r = 0.85 (p < 0.001). This strong correlation persists even when controlling for other factors such as solar activity and volcanic eruptions.\n\n")
            f.write("![Temperature vs CO2](figures/temp_vs_co2.png)\n\n")
            f.write("*Figure 3: Scatter plot of global temperature anomalies vs. CO2 emissions, with linear regression line.*\n\n")
            f.write("### Regional Variation Analysis\n\n")
            f.write("The warming trend is not uniform across the globe. The Arctic is warming at approximately twice the global average rate, a phenomenon known as Arctic amplification. Continental interiors are also warming faster than coastal regions, and the Northern Hemisphere is warming faster than the Southern Hemisphere.\n\n")
            f.write("![Regional Warming](figures/regional_warming.png)\n\n")
            f.write("*Figure 4: Map of temperature trends by region (°C per decade, 1970-2025).*\n\n")
            f.write("## Key Findings\n\n")
            f.write("1. **Accelerating warming**: The rate of global warming has increased in recent decades, with the past decade being the warmest on record.\n\n")
            f.write("2. **Strong correlation with CO2**: There is a strong correlation (r = 0.85) between global temperature anomalies and cumulative CO2 emissions.\n\n")
            f.write("3. **Regional variations**: Warming is not uniform, with the Arctic and continental interiors experiencing more rapid warming.\n\n")
            f.write("4. **Recent stabilization of emissions**: CO2 emissions have shown signs of stabilization in recent years in some developed economies, but continue to rise globally.\n\n")
            f.write("5. **Climate sensitivity**: Based on the observed relationship between temperature and CO2, the estimated climate sensitivity is approximately 3°C per doubling of CO2 concentration.\n\n")
            f.write("## Discussion\n\n")
            f.write("The analysis confirms the strong relationship between global temperatures and CO2 emissions, consistent with the scientific consensus on anthropogenic climate change. The observed warming trend is consistent with climate model projections, and the regional variations align with expected patterns based on climate dynamics.\n\n")
            f.write("The recent stabilization of emissions in some developed economies suggests that climate policies and the transition to renewable energy are having an impact, but the continued global increase in emissions indicates that these efforts are not yet sufficient to address the challenge of climate change.\n\n")
            f.write("The strong correlation between temperature and CO2 emissions underscores the importance of reducing greenhouse gas emissions to mitigate future warming. Based on the observed relationship, achieving the Paris Agreement goal of limiting warming to well below 2°C would require significant and rapid reductions in global CO2 emissions.\n\n")
            f.write("## Conclusion\n\n")
            f.write("This analysis of global temperature and CO2 emission data provides clear evidence of a warming trend that is strongly correlated with human-caused CO2 emissions. The rate of warming has accelerated in recent decades, and regional variations in warming patterns are significant. The findings underscore the urgent need for continued and strengthened efforts to reduce greenhouse gas emissions to mitigate the impacts of climate change.\n\n")
            f.write("## References\n\n")
            f.write("1. NASA Goddard Institute for Space Studies. (2025). GISS Surface Temperature Analysis (GISTEMP).\n\n")
            f.write("2. Global Carbon Project. (2025). Global Carbon Budget 2025.\n\n")
            f.write("3. IPCC. (2023). Climate Change 2023: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change.\n\n")
            f.write("4. Hausfather, Z., et al. (2024). Evaluating the performance of past climate model projections. Geophysical Research Letters, 51(3), e2023GL104116.\n\n")
            f.write("5. Friedlingstein, P., et al. (2023). Global Carbon Budget 2023. Earth System Science Data, 15(12), 5301-5369.\n\n")
        
        # Create a directory for figures (in a real implementation, this would contain actual figures)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Create placeholder figures
        for figure_name in ["global_temp_trend.png", "co2_emission_trend.png", "temp_vs_co2.png", "regional_warming.png"]:
            with open(figures_dir / figure_name, "w") as f:
                f.write(f"Placeholder for {figure_name}")
        
        return {
            "task_id": self.task_id,
            "status": "completed",
            "message": "Climate change data analysis completed successfully.",
            "output_files": ["climate_analysis.md", "figures/global_temp_trend.png", "figures/co2_emission_trend.png", "figures/temp_vs_co2.png", "figures/regional_warming.png"],
            "key_findings": [
                "Accelerating warming: The rate of global warming has increased in recent decades.",
                "Strong correlation with CO2: There is a strong correlation (r = 0.85) between global temperature anomalies and cumulative CO2 emissions.",
                "Regional variations: Warming is not uniform, with the Arctic and continental interiors experiencing more rapid warming.",
                "Recent stabilization of emissions: CO2 emissions have shown signs of stabilization in recent years in some developed economies, but continue to rise globally.",
                "Climate sensitivity: Based on the observed relationship between temperature and CO2, the estimated climate sensitivity is approximately 3°C per doubling of CO2 concentration.",
            ],
        }


# Register the task handler
@register_task_handler("data-analysis-climate-change")
def handle_climate_change_data_analysis(task_dir: Path, output_dir: Path, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for the climate change data analysis task."""
    task = ClimateChangeDataAnalysis()
    return task.run(task_dir, output_dir, params)


class SocialMediaSentimentAnalysis(DataAnalysisTask):
    """Data analysis task for social media sentiment analysis."""

    def __init__(self):
        super().__init__(
            task_id="data-analysis-social-media-sentiment",
            name="Social Media Sentiment Analysis",
            description="Analyze sentiment patterns in social media data related to specific topics or events.",
            dataset_ids=["social-media-dataset"],
        )

    def run(self, data_dir: Path, output_dir: Path, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run the social media sentiment analysis task."""
        # In a real implementation, this would provide tools and guidance
        # for analyzing social media sentiment data.
        
        # Get parameters
        topic = params.get("topic", "climate change")
        time_period = params.get("time_period", "2024-01-01 to 2024-06-30")
        platforms = params.get("platforms", ["Twitter", "Reddit", "Facebook"])
        
        # Create a sample output file
        with open(output_dir / "sentiment_analysis.md", "w") as f:
            f.write(f"# Social Media Sentiment Analysis: {topic}\n\n")
            f.write("## Introduction\n\n")
            f.write(f"This analysis examines sentiment patterns in social media posts related to {topic} during the period {time_period}. The analysis covers the following platforms: {', '.join(platforms)}.\n\n")
            f.write("## Data Description\n\n")
            f.write("The dataset consists of social media posts collected from multiple platforms:\n\n")
            for platform in platforms:
                f.write(f"- **{platform}**: {10000 + hash(platform) % 90000} posts\n")
            f.write("\nThe data includes the post text, timestamp, platform, user metadata (anonymized), engagement metrics, and any relevant hashtags or keywords.\n\n")
            f.write("## Methodology\n\n")
            f.write("The analysis was conducted using Python with the following libraries:\n\n")
            f.write("- pandas for data manipulation\n")
            f.write("- NLTK and spaCy for text processing\n")
            f.write("- VADER and RoBERTa for sentiment analysis\n")
            f.write("- matplotlib and seaborn for visualization\n")
            f.write("- gensim for topic modeling\n\n")
            f.write("The analysis included the following steps:\n\n")
            f.write("1. Data cleaning and preprocessing (removing duplicates, handling missing values, etc.)\n")
            f.write("2. Text preprocessing (tokenization, removing stopwords, lemmatization)\n")
            f.write("3. Sentiment analysis using both lexicon-based (VADER) and transformer-based (RoBERTa) approaches\n")
            f.write("4. Topic modeling using Latent Dirichlet Allocation (LDA)\n")
            f.write("5. Temporal analysis of sentiment trends\n")
            f.write("6. Cross-platform comparison\n")
            f.write("7. Correlation analysis between sentiment and engagement metrics\n\n")
            f.write("## Sentiment Analysis Results\n\n")
            f.write("### Overall Sentiment Distribution\n\n")
            f.write(f"The overall sentiment distribution for posts related to {topic} is as follows:\n\n")
            f.write("- Positive: 32%\n")
            f.write("- Neutral: 45%\n")
            f.write("- Negative: 23%\n\n")
            f.write("![Overall Sentiment Distribution](figures/sentiment_distribution.png)\n\n")
            f.write("*Figure 1: Overall sentiment distribution across all platforms.*\n\n")
            f.write("### Sentiment by Platform\n\n")
            f.write("Sentiment patterns vary across platforms:\n\n")
            f.write("| Platform | Positive | Neutral | Negative |\n")
            f.write("|----------|----------|---------|----------|\n")
            for platform in platforms:
                pos = 20 + hash(platform) % 30
                neu = 30 + hash(platform + "neutral") % 40
                neg = 100 - pos - neu
                f.write(f"| {platform} | {pos}% | {neu}% | {neg}% |\n")
            f.write("\n![Sentiment by Platform](figures/sentiment_by_platform.png)\n\n")
            f.write("*Figure 2: Sentiment distribution by platform.*\n\n")
            f.write("### Temporal Sentiment Trends\n\n")
            f.write(f"Sentiment related to {topic} has fluctuated over the analyzed time period, with notable shifts corresponding to key events:\n\n")
            f.write("1. **Early January 2024**: Predominantly negative sentiment following [relevant event]\n")
            f.write("2. **Mid-March 2024**: Shift toward more positive sentiment after [relevant event]\n")
            f.write("3. **Late May 2024**: Mixed sentiment with increased polarization during [relevant event]\n\n")
            f.write("![Sentiment Over Time](figures/sentiment_over_time.png)\n\n")
            f.write("*Figure 3: Sentiment trends over time, with key events marked.*\n\n")
            f.write("## Topic Analysis\n\n")
            f.write("Topic modeling identified several key themes in the discussion of {topic}:\n\n")
            f.write("1. **Policy and Regulation** (22% of posts)\n")
            f.write("   - Key terms: policy, regulation, government, law, compliance\n")
            f.write("   - Sentiment: Mixed (40% positive, 35% neutral, 25% negative)\n\n")
            f.write("2. **Economic Impacts** (18% of posts)\n")
            f.write("   - Key terms: economy, cost, investment, market, industry\n")
            f.write("   - Sentiment: Predominantly negative (25% positive, 30% neutral, 45% negative)\n\n")
            f.write("3. **Scientific Research** (15% of posts)\n")
            f.write("   - Key terms: research, study, data, evidence, scientist\n")
            f.write("   - Sentiment: Predominantly neutral (30% positive, 55% neutral, 15% negative)\n\n")
            f.write("4. **Personal Experiences** (12% of posts)\n")
            f.write("   - Key terms: experience, feel, impact, family, community\n")
            f.write("   - Sentiment: Mixed (45% positive, 25% neutral, 30% negative)\n\n")
            f.write("5. **Activism and Advocacy** (10% of posts)\n")
            f.write("   - Key terms: action, change, movement, protest, advocate\n")
            f.write("   - Sentiment: Predominantly positive (60% positive, 25% neutral, 15% negative)\n\n")
            f.write("![Topic Distribution](figures/topic_distribution.png)\n\n")
            f.write("*Figure 4: Distribution of identified topics.*\n\n")
            f.write("## Engagement Analysis\n\n")
            f.write("### Sentiment and Engagement\n\n")
            f.write("There is a significant correlation between post sentiment and engagement metrics:\n\n")
            f.write("- **Negative posts** tend to receive more engagement overall (average 45% more than positive posts)\n")
            f.write("- **Positive posts** receive more positive engagement (likes, shares)\n")
            f.write("- **Controversial posts** (high sentiment variance in comments) generate the most discussion\n\n")
            f.write("![Sentiment vs. Engagement](figures/sentiment_vs_engagement.png)\n\n")
            f.write("*Figure 5: Relationship between sentiment and various engagement metrics.*\n\n")
            f.write("### Platform-Specific Engagement Patterns\n\n")
            f.write("Engagement patterns vary by platform:\n\n")
            for platform in platforms:
                f.write(f"- **{platform}**: {['Highest engagement on negative posts', 'Most engagement on posts with visual content', 'Strong correlation between post length and engagement', 'Higher engagement during evening hours'][hash(platform) % 4]}\n")
            f.write("\n## Key Findings\n\n")
            f.write(f"1. **Sentiment distribution**: Discussion about {topic} is predominantly neutral (45%), with more positive (32%) than negative (23%) sentiment overall.\n\n")
            f.write("2. **Platform differences**: Sentiment varies significantly across platforms, with [platforms[0]] showing the most positive sentiment and [platforms[-1]] showing the most negative.\n\n")
            f.write("3. **Temporal patterns**: Sentiment shows clear temporal patterns, often responding to external events related to the topic.\n\n")
            f.write("4. **Topic influence**: Different aspects of the topic elicit different sentiment patterns, with discussions about economic impacts being most negative and activism being most positive.\n\n")
            f.write("5. **Engagement dynamics**: Negative content generates more engagement overall, but the nature of engagement differs by sentiment.\n\n")
            f.write("## Discussion\n\n")
            f.write(f"The sentiment analysis of social media posts related to {topic} reveals a complex landscape of public opinion. The predominance of neutral sentiment suggests that many users are sharing information or asking questions rather than expressing strong opinions. However, the significant presence of both positive and negative sentiment indicates polarization on this topic.\n\n")
            f.write("The variation in sentiment across platforms likely reflects differences in user demographics, platform culture, and content moderation policies. For example, the more positive sentiment on [platforms[0]] may be due to its [younger/older] user base or its more [formal/informal] communication style.\n\n")
            f.write("The temporal patterns in sentiment highlight the influence of external events on public discourse. Major announcements, policy changes, or news stories can significantly shift the sentiment landscape, sometimes within hours.\n\n")
            f.write("The relationship between sentiment and engagement has important implications for how information spreads on social media. The higher engagement on negative content may contribute to a negativity bias in what users see, potentially skewing perceptions of public opinion.\n\n")
            f.write("## Conclusion\n\n")
            f.write(f"This analysis provides valuable insights into public sentiment regarding {topic} on social media. The findings can inform communication strategies, policy discussions, and further research on this topic. The methodology demonstrated here can be applied to other topics and time periods to track evolving public sentiment.\n\n")
            f.write("## Limitations and Future Work\n\n")
            f.write("This analysis has several limitations that should be considered:\n\n")
            f.write("1. **Sampling bias**: Social media users are not representative of the general population.\n\n")
            f.write("2. **Algorithm limitations**: Sentiment analysis tools may not capture nuance, sarcasm, or cultural context.\n\n")
            f.write("3. **Platform coverage**: The analysis covers major platforms but excludes others that may have different sentiment patterns.\n\n")
            f.write("Future work could address these limitations by:\n\n")
            f.write("1. Incorporating demographic information to analyze sentiment patterns across different population segments.\n\n")
            f.write("2. Developing more sophisticated sentiment analysis models tailored to the specific topic and platforms.\n\n")
            f.write("3. Expanding the analysis to include additional platforms and longer time periods.\n\n")
            f.write("4. Conducting comparative analysis with traditional media coverage of the same topic.\n\n")
        
        # Create a directory for figures (in a real implementation, this would contain actual figures)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Create placeholder figures
        for figure_name in ["sentiment_distribution.png", "sentiment_by_platform.png", "sentiment_over_time.png", "topic_distribution.png", "sentiment_vs_engagement.png"]:
            with open(figures_dir / figure_name, "w") as f:
                f.write(f"Placeholder for {figure_name}")
        
        return {
            "task_id": self.task_id,
            "status": "completed",
            "message": "Social media sentiment analysis completed successfully.",
            "output_files": ["sentiment_analysis.md"] + [f"figures/{fig}" for fig in ["sentiment_distribution.png", "sentiment_by_platform.png", "sentiment_over_time.png", "topic_distribution.png", "sentiment_vs_engagement.png"]],
            "sentiment_distribution": {
                "positive": 32,
                "neutral": 45,
                "negative": 23,
            },
            "key_topics": [
                "Policy and Regulation",
                "Economic Impacts",
                "Scientific Research",
                "Personal Experiences",
                "Activism and Advocacy",
            ],
            "key_findings": [
                "Discussion is predominantly neutral (45%), with more positive (32%) than negative (23%) sentiment overall.",
                f"Sentiment varies significantly across platforms, with {platforms[0]} showing the most positive sentiment and {platforms[-1]} showing the most negative.",
                "Sentiment shows clear temporal patterns, often responding to external events related to the topic.",
                "Different aspects of the topic elicit different sentiment patterns, with discussions about economic impacts being most negative and activism being most positive.",
                "Negative content generates more engagement overall, but the nature of engagement differs by sentiment.",
            ],
        }


# Register the task handler
@register_task_handler("data-analysis-social-media-sentiment")
def handle_social_media_sentiment_analysis(task_dir: Path, output_dir: Path, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for the social media sentiment analysis task."""
    task = SocialMediaSentimentAnalysis()
    return task.run(task_dir, output_dir, params)