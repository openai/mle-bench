import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import tempfile
import shutil
import os
import json
from omegaconf import OmegaConf
from rich.console import Console
import sys
from dotenv import load_dotenv
import logging
from aide import Experiment

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

logger = logging.getLogger("aide")
logger.setLevel(logging.INFO)

console = Console(file=sys.stderr)


class WebUI:
    """
    WebUI encapsulates the Streamlit application logic for the AIDE Machine Learning Engineer Agent.
    """

    def __init__(self):
        """
        Initialize the WebUI with environment variables and session state.
        """
        self.env_vars = self.load_env_variables()
        self.project_root = Path(__file__).parent.parent.parent
        self.config_session_state()
        self.setup_page()

    @staticmethod
    def load_env_variables():
        """
        Load API keys and environment variables from .env file.

        Returns:
            dict: Dictionary containing API keys.
        """
        load_dotenv()
        return {
            "openai_key": os.getenv("OPENAI_API_KEY", ""),
            "anthropic_key": os.getenv("ANTHROPIC_API_KEY", ""),
            "gemini_key": os.getenv("GEMINI_API_KEY", ""),
            "openrouter_key": os.getenv("OPENROUTER_API_KEY", ""),
        }

    @staticmethod
    def config_session_state():
        """
        Configure default values for Streamlit session state.
        """
        if "is_running" not in st.session_state:
            st.session_state.is_running = False
        if "current_step" not in st.session_state:
            st.session_state.current_step = 0
        if "total_steps" not in st.session_state:
            st.session_state.total_steps = 0
        if "progress" not in st.session_state:
            st.session_state.progress = 0
        if "results" not in st.session_state:
            st.session_state.results = None

    @staticmethod
    def setup_page():
        """
        Set up the Streamlit page configuration and load custom CSS.
        """
        st.set_page_config(
            page_title="AIDE: Machine Learning Engineer Agent",
            layout="wide",
        )
        WebUI.load_css()

    @staticmethod
    def load_css():
        """
        Load custom CSS styles from 'style.css' file.
        """
        css_file = Path(__file__).parent / "style.css"
        if css_file.exists():
            with open(css_file) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        else:
            st.warning(f"CSS file not found at: {css_file}")

    def run(self):
        """
        Run the main logic of the Streamlit application.
        """
        self.render_sidebar()
        input_col, results_col = st.columns([1, 3])
        with input_col:
            self.render_input_section(results_col)

    def render_sidebar(self):
        """
        Render the sidebar with API key settings.
        """
        with st.sidebar:
            st.header("⚙️ Settings")
            st.markdown(
                "<p style='text-align: center;'>OpenAI API Key</p>",
                unsafe_allow_html=True,
            )
            openai_key = st.text_input(
                "OpenAI API Key",
                value=self.env_vars["openai_key"],
                type="password",
                label_visibility="collapsed",
            )
            st.markdown(
                "<p style='text-align: center;'>Anthropic API Key</p>",
                unsafe_allow_html=True,
            )
            anthropic_key = st.text_input(
                "Anthropic API Key",
                value=self.env_vars["anthropic_key"],
                type="password",
                label_visibility="collapsed",
            )
            st.markdown(
                "<p style='text-align: center;'>OpenRouter API Key</p>",
                unsafe_allow_html=True,
            )
            openrouter_key = st.text_input(
                "OpenRouter API Key",
                value=self.env_vars["openrouter_key"],
                type="password",
                label_visibility="collapsed",
            )
            if st.button("Save API Keys", use_container_width=True):
                st.session_state.openai_key = openai_key
                st.session_state.anthropic_key = anthropic_key
                st.session_state.openrouter_key = openrouter_key
                st.success("API keys saved!")

    def render_input_section(self, results_col):
        """
        Render the input section of the application.

        Args:
            results_col (st.delta_generator.DeltaGenerator): The results column to pass to methods.
        """
        st.header("Input")
        uploaded_files = self.handle_file_upload()
        goal_text, eval_text, num_steps = self.handle_user_inputs()
        if st.button("Run AIDE", type="primary", use_container_width=True):
            with st.spinner("AIDE is running..."):
                results = self.run_aide(
                    uploaded_files, goal_text, eval_text, num_steps, results_col
                )
                st.session_state.results = results

    def handle_file_upload(self):
        """
        Handle file uploads and example file loading.

        Returns:
            list: List of uploaded or example files.
        """
        # Only show file uploader if no example files are loaded
        if not st.session_state.get("example_files"):
            uploaded_files = st.file_uploader(
                "Upload Data Files",
                accept_multiple_files=True,
                type=["csv", "txt", "json", "md"],
                label_visibility="collapsed",
            )

            if uploaded_files:
                st.session_state.pop(
                    "example_files", None
                )  # Remove example files if any
                return uploaded_files

            # Only show example button if no files are uploaded
            if st.button(
                "Load Example Experiment", type="primary", use_container_width=True
            ):
                st.session_state.example_files = self.load_example_files()

        if st.session_state.get("example_files"):
            st.info("Example files loaded! Click 'Run AIDE' to proceed.")
            with st.expander("View Loaded Files", expanded=False):
                for file in st.session_state.example_files:
                    st.text(f"📄 {file['name']}")
            return st.session_state.example_files

        return []  # Return empty list if no files are uploaded or loaded

    def handle_user_inputs(self):
        """
        Handle goal, evaluation criteria, and number of steps inputs.

        Returns:
            tuple: Goal text, evaluation criteria text, and number of steps.
        """
        goal_text = st.text_area(
            "Goal",
            value=st.session_state.get("goal", ""),
            placeholder="Example: Predict the sales price for each house",
        )
        eval_text = st.text_area(
            "Evaluation Criteria",
            value=st.session_state.get("eval", ""),
            placeholder="Example: Use the RMSE metric between the logarithm of the predicted and observed values.",
        )
        num_steps = st.slider(
            "Number of Steps",
            min_value=1,
            max_value=20,
            value=st.session_state.get("steps", 10),
        )
        return goal_text, eval_text, num_steps

    @staticmethod
    def load_example_files():
        """
        Load example files from the 'example_tasks/house_prices' directory.

        Returns:
            list: List of example files with their paths.
        """
        package_root = Path(__file__).parent.parent
        example_dir = package_root / "example_tasks" / "house_prices"

        if not example_dir.exists():
            st.error(f"Example directory not found at: {example_dir}")
            return []

        example_files = []

        for file_path in example_dir.glob("*"):
            if file_path.suffix.lower() in [".csv", ".txt", ".json", ".md"]:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_path.suffix
                ) as tmp_file:
                    tmp_file.write(file_path.read_bytes())
                    example_files.append(
                        {"name": file_path.name, "path": tmp_file.name}
                    )

        if not example_files:
            st.warning("No example files found in the example directory")

        st.session_state["goal"] = "Predict the sales price for each house"
        st.session_state["eval"] = (
            "Use the RMSE metric between the logarithm of the predicted and observed values."
        )

        return example_files

    def run_aide(self, files, goal_text, eval_text, num_steps, results_col):
        """
        Run the AIDE experiment with the provided inputs.

        Args:
            files (list): List of uploaded or example files.
            goal_text (str): The goal of the experiment.
            eval_text (str): The evaluation criteria.
            num_steps (int): Number of steps to run.
            results_col (st.delta_generator.DeltaGenerator): Results column for displaying progress.

        Returns:
            dict: Dictionary containing the results of the experiment.
        """
        try:
            self.initialize_run_state(num_steps)
            self.set_api_keys()

            input_dir = self.prepare_input_directory(files)
            if not input_dir:
                return None

            experiment = self.initialize_experiment(input_dir, goal_text, eval_text)

            # Create separate placeholders for progress and config
            progress_placeholder = results_col.empty()
            config_placeholder = results_col.empty()
            results_placeholder = results_col.empty()

            for step in range(num_steps):
                st.session_state.current_step = step + 1
                progress = (step + 1) / num_steps

                # Update progress
                with progress_placeholder.container():
                    st.markdown(
                        f"### 🔥 Running Step {st.session_state.current_step}/{st.session_state.total_steps}"
                    )
                    st.progress(progress)

                # Show config only for first step
                if step == 0:
                    with config_placeholder.container():
                        st.markdown("### 📋 Configuration")
                        st.code(OmegaConf.to_yaml(experiment.cfg), language="yaml")

                experiment.run(steps=1)

                # Show results
                with results_placeholder.container():
                    self.render_live_results(experiment)

                # Clear config after first step
                if step == 0:
                    config_placeholder.empty()

            # Clear progress after all steps
            progress_placeholder.empty()

            # Update session state
            st.session_state.is_running = False
            st.session_state.results = self.collect_results(experiment)
            return st.session_state.results

        except Exception as e:
            st.session_state.is_running = False
            console.print_exception()
            st.error(f"Error occurred: {str(e)}")
            return None

    @staticmethod
    def initialize_run_state(num_steps):
        """
        Initialize the running state for the experiment.

        Args:
            num_steps (int): Total number of steps in the experiment.
        """
        st.session_state.is_running = True
        st.session_state.current_step = 0
        st.session_state.total_steps = num_steps
        st.session_state.progress = 0

    @staticmethod
    def set_api_keys():
        """
        Set the API keys in the environment variables from the session state.
        """
        if st.session_state.get("openai_key"):
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
        if st.session_state.get("anthropic_key"):
            os.environ["ANTHROPIC_API_KEY"] = st.session_state.anthropic_key
        if st.session_state.get("gemini_key"):
            os.environ["GEMINI_API_KEY"] = st.session_state.gemini_key
        if st.session_state.get("openrouter_key"):
            os.environ["OPENROUTER_API_KEY"] = st.session_state.openrouter_key

    def prepare_input_directory(self, files):
        """
        Prepare the input directory and handle uploaded files.

        Args:
            files (list): List of uploaded or example files.

        Returns:
            Path: The input directory path, or None if files are missing.
        """
        input_dir = self.project_root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        if files:
            for file in files:
                if isinstance(file, dict):  # Example files
                    shutil.copy2(file["path"], input_dir / file["name"])
                else:  # Uploaded files
                    with open(input_dir / file.name, "wb") as f:
                        f.write(file.getbuffer())
        else:
            st.error("Please upload data files")
            return None
        return input_dir

    @staticmethod
    def initialize_experiment(input_dir, goal_text, eval_text):
        """
        Initialize the AIDE Experiment.

        Args:
            input_dir (Path): Path to the input directory.
            goal_text (str): The goal of the experiment.
            eval_text (str): The evaluation criteria.

        Returns:
            Experiment: The initialized Experiment object.
        """
        experiment = Experiment(data_dir=str(input_dir), goal=goal_text, eval=eval_text)
        return experiment

    @staticmethod
    def collect_results(experiment):
        """
        Collect the results from the experiment.

        Args:
            experiment (Experiment): The Experiment object.

        Returns:
            dict: Dictionary containing the collected results.
        """
        solution_path = experiment.cfg.log_dir / "best_solution.py"
        if solution_path.exists():
            solution = solution_path.read_text()
        else:
            solution = "No solution found"

        journal_data = [
            {
                "step": node.step,
                "code": str(node.code),
                "metric": str(node.metric.value) if node.metric else None,
                "is_buggy": node.is_buggy,
            }
            for node in experiment.journal.nodes
        ]

        results = {
            "solution": solution,
            "config": OmegaConf.to_yaml(experiment.cfg),
            "journal": json.dumps(journal_data, indent=2, default=str),
            "tree_path": str(experiment.cfg.log_dir / "tree_plot.html"),
        }
        return results

    @staticmethod
    def render_tree_visualization(results):
        """
        Render the tree visualization from the experiment results.

        Args:
            results (dict): The results dictionary containing paths and data.
        """
        if "tree_path" in results:
            tree_path = Path(results["tree_path"])
            logger.info(f"Loading tree visualization from: {tree_path}")
            if tree_path.exists():
                with open(tree_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                components.html(html_content, height=600, scrolling=True)
            else:
                st.error(f"Tree visualization file not found at: {tree_path}")
                logger.error(f"Tree file not found at: {tree_path}")
        else:
            st.info("No tree visualization available for this run.")

    @staticmethod
    def render_best_solution(results):
        """
        Display the best solution code.

        Args:
            results (dict): The results dictionary containing the solution.
        """
        if "solution" in results:
            solution_code = results["solution"]
            st.code(solution_code, language="python")
        else:
            st.info("No solution available.")

    @staticmethod
    def render_config(results):
        """
        Display the configuration used in the experiment.

        Args:
            results (dict): The results dictionary containing the config.
        """
        if "config" in results:
            st.code(results["config"], language="yaml")
        else:
            st.info("No configuration available.")

    @staticmethod
    def render_journal(results):
        """
        Display the experiment journal as JSON.

        Args:
            results (dict): The results dictionary containing the journal.
        """
        if "journal" in results:
            try:
                journal_data = json.loads(results["journal"])
                formatted_journal = json.dumps(journal_data, indent=2)
                st.code(formatted_journal, language="json")
            except json.JSONDecodeError:
                st.code(results["journal"], language="json")
        else:
            st.info("No journal available.")

    @staticmethod
    def get_best_metric(results):
        """
        Extract the best validation metric from results.
        """
        try:
            journal_data = json.loads(results["journal"])
            metrics = []
            for node in journal_data:
                if node["metric"] is not None:
                    try:
                        # Convert string metric to float
                        metric_value = float(node["metric"])
                        metrics.append(metric_value)
                    except (ValueError, TypeError):
                        continue
            return max(metrics) if metrics else None
        except (json.JSONDecodeError, KeyError):
            return None

    @staticmethod
    def render_validation_plot(results, step):
        """
        Render the validation score plot.

        Args:
            results (dict): The results dictionary
            step (int): Current step number for unique key generation
        """
        try:
            journal_data = json.loads(results["journal"])
            steps = []
            metrics = []

            for node in journal_data:
                if node["metric"] is not None and node["metric"].lower() != "none":
                    try:
                        metric_value = float(node["metric"])
                        steps.append(node["step"])
                        metrics.append(metric_value)
                    except (ValueError, TypeError):
                        continue

            if metrics:
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=metrics,
                        mode="lines+markers",
                        name="Validation Score",
                        line=dict(color="#F04370"),
                        marker=dict(color="#F04370"),
                    )
                )

                fig.update_layout(
                    title="Validation Score Progress",
                    xaxis_title="Step",
                    yaxis_title="Validation Score",
                    template="plotly_white",
                    hovermode="x unified",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )

                # Only keep the key for plotly_chart
                st.plotly_chart(fig, use_container_width=True, key=f"plot_{step}")
            else:
                st.info("No validation metrics available to plot")

        except (json.JSONDecodeError, KeyError):
            st.error("Could not parse validation metrics data")

    def render_live_results(self, experiment):
        """
        Render live results.

        Args:
            experiment (Experiment): The Experiment object
        """
        results = self.collect_results(experiment)

        # Create tabs for different result views
        tabs = st.tabs(
            [
                "Tree Visualization",
                "Best Solution",
                "Config",
                "Journal",
                "Validation Plot",
            ]
        )

        with tabs[0]:
            self.render_tree_visualization(results)
        with tabs[1]:
            self.render_best_solution(results)
        with tabs[2]:
            self.render_config(results)
        with tabs[3]:
            self.render_journal(results)
        with tabs[4]:
            best_metric = self.get_best_metric(results)
            if best_metric is not None:
                st.metric("Best Validation Score", f"{best_metric:.4f}")
            self.render_validation_plot(results, step=st.session_state.current_step)


if __name__ == "__main__":
    app = WebUI()
    app.run()
