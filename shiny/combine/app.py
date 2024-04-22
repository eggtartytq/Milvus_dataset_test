from shiny import App, render, ui, reactive
# from shiny.express import input
from BGE_search import BGE_search
from Sentrans_search import Sentrans_search
from Splade_search import Splade_search
import pandas as pd
import logging

my_css = """
<style>

table th {
    text-align: left;
}
</style>
"""
emb_lst = ["BGE_search", "Sentrans_search", "Splade_search"]
emb_func_map ={
    "BGE_search" : BGE_search,
    "Sentrans_search" : Sentrans_search,
    "Splade_search": Splade_search,
 }
app_ui = ui.page_fluid(
    ui.tags.head(ui.HTML(my_css)),
    ui.row(
        
        ui.column(
            3,
            ui.input_text("txt_input", "Input"),
            ui.input_numeric("result_num", "Limit", value=10, min=1, max=50),
            ui.input_selectize("emb_list", "Embedding Function", choices = emb_lst, selected=emb_lst[0]),
            ui.input_checkbox_group(
                "header_list",
                "Attributes",
                {
                    "Data_Id" : "Data_Id",
                    # "distance" : "distance",
                    "Dataset_File" : "Dataset_File",
                    "Collected" : "Collected",
                    "Variable" : "Variable",
                    "Label" : "Label",
                    "Description" : "Description"
                },
                selected=['Data_Id', 'Label', 'Description']
            ),
            ui.input_action_button("btn_search", "Search"),
            ui.input_action_button("btn_reset", "Reset")
        ),
        
        ui.column(
            9,
            ui.output_ui("intro_milvus"),
            ui.output_ui("search_result_card")
        )
    )
)



def server(input, output, session):
    
    prev_search_count = reactive.Value(0)
    show_result = reactive.Value(False)

    @reactive.Effect
    def check_search_card():
        if input.btn_search()>0:
            show_result.set(True)
    
    @reactive.Effect
    @reactive.event(input.btn_reset)
    def check_reset():
        if input.btn_reset() > 0:
            show_result.set(False)
            # session.set_input("txt_input", " ")
            

    @output
    @render.ui
    def search_result_card():
        if show_result():
            return ui.card(ui.output_table("search_result"))
        else:
            return ui.div(style="display: none;")

    #hide the introduction when user click search
    @output
    @render.ui
    def intro_milvus():
        if not show_result():
            return ui.div(
                            # ui.div(
                            #     ui.img(src="https://www.google.com/imgres?q=milvus&imgurl=https%3A%2F%2Fmilvus.io%2Fstatic%2F0bc2e74d0a1b20bbfb91bdbd03f77e5e%2Fbbbf7%2Farchitecture_diagram.png&imgrefurl=https%3A%2F%2Fmilvus.io%2Fdocs%2Fv2.0.x%2Farchitecture_overview.md&docid=Xq5NScqwxSYdcM&tbnid=y8CUfem7vlYJUM&vet=12ahUKEwjM_8Thw6WFAxWmJzQIHfLLAZoQM3oECEMQAA..i&w=1280&h=903&hcb=2&ved=2ahUKEwjM_8Thw6WFAxWmJzQIHfLLAZoQM3oECEMQAA")
                            # ),
                            ui.div(
                                ui.h1("About searching"),
                                ui.p("The program will transform the user's input into a vector for the purpose of performing a similarity search within the database data."),
                                ui.h2("Quick start"),
                                ui.p("1. Enter the content user want to search"),
                                ui.p("2. Enter the desired number of results (1~10)"),
                                ui.p("3. Select the attributes shown in the resu"),
                                ui.p("4. click Search Button to search"),
                                ui.p("5. click Reset to reset the searching page"),
                                ui.h3("Input Box"),
                                ui.p("Users input the content they wish to search into the input box."),
                                ui.h3("Limit Box"),
                                ui.p("Limit the number of search results to an integer between 1 and 10, with the default set to 5."),
                                ui.h3("Attributes"),
                                ui.p("The attributes for the searching result"),
                                ui.h3("Search Button"),
                                ui.p("The program convert user's input into vector and perform a simlarity search"),
                                ui.h3("Reset Button"),
                                ui.p("Clear the searching result"),
                            ),
                            ui.div(
                                ui.tags.style("""
                                        .table_header {
                                            background-color: #f2f2f2; /* 浅灰色背景 */
                                            padding: 8px; /* 内边距 */
                                            border-bottom: 1px solid #ddd; /* 底部边框 */
                                            font-weight: bold; /* 加粗字体 */
                                            text-align: left; /* 文本左对齐 */
                                            flex-grow: 1; /* 使所有列宽相同 */
                                        }
        
                                        .flex-container {
                                            display: flex; /* 启用flex布局 */
                                            flex-direction: row; /* 项目水平排列 */
                                        }

                                """),
                                ui.div(
                                ui.h1("Example dataset"),
                                ui.div(
                                    ui.div(
                                        ui.div("Data_Id", class_="table_header"),
                                        ui.div("Dataset_File", class_="table_header"),
                                        ui.div("Collected", class_="table_header"),
                                        ui.div("Variable", class_="table_header"),
                                        ui.div("Label", class_="table_header"),
                                        ui.div("Description", class_="table_header"),
                                        ui.div("Info_embedding", class_="table_header"),
                                        class_="flex-container"
                                    ),
                                )
                            )
                            ),
                            ui.div(
                                ui.h3("Emdedding Function"),
                                ui.p("Sentence Transformer: Dense, Splade: Sparse, BGE-M3 : Hybrid")
                                
                            )
                            
                            
                          )
        else:
            return ui.div(style='dispaly: none;')

    #search result
    @output
    @render.table
    @reactive.event(input.btn_search, ignore_none=False)
    def search_result():
        if show_result():
            try:
                print(f"{input.btn_search()} VS {prev_search_count()}")

                search_query = input.txt_input()
                search_obj = emb_func_map[input.emb_list()](limit=input.result_num())
                result = search_obj.embed_search(search_query, input.header_list())
                prev_search_count.set(input.btn_search())
                return result
            except Exception as e:
                return pd.DataFrame({"Error": [str(e)]})
            
            

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
