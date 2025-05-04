
def PromptGeneration(final_table, image_associated_columns, question, metadata):
    final_table = final_table.rename(columns={'ImagePath':image_associated_columns[0]+' (Image)'})
    # 400
    example_1 = """Question: What was Cristiano Ronaldo's club when he received Top assists in the 2018-19 Serie A?

                Rank: Table Text
                Player: Table Text
                Club: Table Text
                Assists: Table Text
                Player (Image): Image
                And Text outside Table

                Composition: TableQ
                TableQ:  What was Cristiano Ronaldo's club when he received Top assists in the 2018-19 Serie A"""
    # 800
    example_2 = """Question: What is the man holding in the poster for Blue Eyes (film)?

                Actor: Table Text
                Role: Table Text
                Actor (Image): Image
                And Text outside Table

                Composition: ImageQ
                ImageQ:  What is the man holding in the poster for Blue Eyes (film)"""
    # 2
    example_3 = """Question: What sports is the Ben Piazza 1976 movie title?

                Year: Table Text
                Title: Table Text
                Role: Table Text
                Notes: Table Text
                Title (Image): Image
                And Text outside Table

                Composition: Compose(ImageQ,TableQ)
                ImageQ: What sports is the Title (Image)?
                TableQ: What is the 1976 movie Title (Image)?"""
    

    # prompt = f'Given the following examples and question compositions\n Example 1: {example_1}\n Example 2:{example_2}\n Example 3:{example_3} \n Decompose the following question given the table, image and text information'
    prompt = f"{example_1}\n {example_2}\n {example_3}\n"
    prompt += f"Question: {question}\n"
    for a_col in final_table.columns:
        if 'Image' not in a_col:
            prompt += f"{a_col}: Table Text\n"
        else:
            prompt += f"{a_col}: Image\n"
    
    prompt += "And Text outside Table\n"


    return prompt

if __name__ == '__main__':
    pass