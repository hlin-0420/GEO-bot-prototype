# Creating a chatbot function

class GeoCurveChatbot:
    def __init__(self, dataframe):
        self.data = dataframe
        self.data.columns = ["Curve_Type", "Description"]  # Renaming columns for easier handling

    def get_curve_info(self, curve_name):
        """
        Retrieve information about a specific curve type.
        """
        result = self.data[self.data["Curve_Type"].str.lower() == curve_name.lower()]
        if not result.empty:
            return result.iloc[0]["Description"]
        else:
            return "Sorry, I couldn't find information on that curve type. Please try another one."