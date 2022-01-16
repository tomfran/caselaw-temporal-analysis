#!/usr/bin/env python
# coding: utf-8


import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import pickle
import numpy
import random
import ast

import warnings
warnings.filterwarnings('ignore')

from dash import Dash
from dash import dcc
from dash import html
from dash import Input, Output, State, ALL
from wordcloud import WordCloud
from plotly.validator_cache import ValidatorCache

from webapp_utility import Loader


PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
TEMPLATE = 'plotly_white'

app_loader = Loader()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# ##### Navbar

NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(
                        dbc.NavbarBrand("Illinois Cases Analysis", className="ml-2"),
                        style={"marginLeft": 10}
                    ),
                ],
                align="center",
                className="g-0",
            ),
            href="https://github.com/tomfran/legal-texts-information-retrieval",
            style={"margin": 10, "textDecoration": "none"}
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)


# ##### Searchbox

SEARCH_BOX = dbc.InputGroup(
    [
        dbc.Button("Search", id="search-button", n_clicks=0),
        dbc.Input(id="search-input", placeholder="cocaine, drug - gun, weapon"),
    ],
    style={"marginTop": 20}
)


# Word Analysis

WORD_DROPDOWN = dcc.Dropdown(id="words-drop", clearable=False, style={"font-size": 12})
CONTEXT_GRAPH = dcc.Loading(
    id="loading-similar-context-words",
    children=[dcc.Graph(id="similar-context-graph")],
    type="default",
)
GRAMS_GRAPH = dcc.Loading(
    id="loading-grams", 
    children=[dcc.Graph(id="grams-graph")],
    type="default",
)
SEMANTIC_YEAR_SLIDER = dcc.Slider(
    id="semantic-year-slider",
    step=1,
    tooltip={"placement": "bottom", "always_visible": True},
)
SEMANTIC_YEARLY_SHIFT_GRAPH = dcc.Loading(
    id="loading-yearly-semantic", 
    children=[dcc.Graph(id="semantic-yearly-shift-graph")],
    type="default",
)
SEMANTIC_YEARLY_FIRST = dcc.Loading(
    id="loading-semantic-yearly-first",
    children=[dcc.Graph(id="semantic-yearly-first-graph")],
    type="default",
)
SEMANTIC_YEARLY_SECOND = dcc.Loading(
    id="loading-semantic-yearly-second",
    children=[dcc.Graph(id="semantic-yearly-second-graph")],
    type="default",
)

SEMANTIC_EPOCH_SHIFT_GRAPH = dcc.Loading(
    id="loading-epoch-semantic", 
    children=[dcc.Graph(id="semantic-epoch-shift-graph")],
    type="default",
)
SEMANTIC_EPOCH_FIRST = dcc.Loading(
    id="loading-semantic-epoch-first",
    children=[dcc.Graph(id="semantic-epoch-first-graph")],
    type="default",
)
SEMANTIC_EPOCH_SECOND = dcc.Loading(
    id="loading-semantic-epoch-second",
    children=[dcc.Graph(id="semantic-epoch-second-graph")],
    type="default",
)

SEMANTIC_SHIFT_TABS = dcc.Tabs(
    id="word-semantic-shift-tabs",
    value="Epoch",
    children=[
        dcc.Tab(
            label="Epoch",
            value='Epoch',
            children=[
                dbc.Row([dbc.Col(SEMANTIC_EPOCH_SHIFT_GRAPH, md=6), dbc.Col(SEMANTIC_EPOCH_FIRST), dbc.Col(SEMANTIC_EPOCH_SECOND)], className="g-0")
            ]
        ),
        dcc.Tab(
            label="Yearly",
            value='Yeary',
            children=[
                SEMANTIC_YEAR_SLIDER,
                dbc.Row([dbc.Col(SEMANTIC_YEARLY_SHIFT_GRAPH, md=6), dbc.Col(SEMANTIC_YEARLY_FIRST), dbc.Col(SEMANTIC_YEARLY_SECOND)], className="g-0")
            ]
        ),
    ]
)

WORD_GENERIC_TOPIC_DISTRIBUTION_GRAPH = dcc.Loading(
    id="loading-word-topics", 
    children=[dcc.Graph(id="word-topics-graph")],
    type="default",
)

TOPICS_LIST = dcc.Loading(
    id="loading-topic-descriptions", 
    children=[dbc.ListGroup(id="topic-descriptions-list", children=[],)],
    type="default",
)

WORD_TOPIC_TABS = dcc.Tabs(
    id="word-topics-tabs",
    value="General",
    children=[
        dcc.Tab(
            label="General",
            value='General'
        ),
        dcc.Tab(
            label="Specific",
            value='Specific'
        ),
    ]
)


WORD_CARD = [
    dbc.CardHeader(html.H5("Word analysis")),
    dbc.Alert(
        "Not enough data to render these plots, please adjust the filters",
        id="no-word-data-alert",
        color="warning",
        style={"display": "none"},
    ),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col([
                        WORD_DROPDOWN,
                        CONTEXT_GRAPH
                    ]),
                    dbc.Col([GRAMS_GRAPH], md=8)
                ],
                className="g-0"
            ),
            SEMANTIC_SHIFT_TABS,
            dbc.Row([
                WORD_TOPIC_TABS, 
                dbc.Col(WORD_GENERIC_TOPIC_DISTRIBUTION_GRAPH), dbc.Col(TOPICS_LIST, md=6)
            ], 
                justify="center", 
                align="center",)
        ]
    )
]


# Topic Analysis

TOPIC_WORDS_GRAPHS = dbc.Row(
    [
        dbc.Col(
            dcc.Loading(
                id="loading-topic-top-words",
                children=[dcc.Graph(id="topic-top-words-graph")],
                type="default",
            )
        ),
        dbc.Col(
            [
                dcc.Tabs(
                    id="tabs",
                    children=[
                        dcc.Tab(
                            label="Wordcloud",
                            children=[
                                dcc.Loading(
                                    id="loading-wordcloud",
                                    children=[
                                        dcc.Graph(id="topic-wordcloud")
                                    ],
                                    type="default",
                                )
                            ],
                        ),
                        dcc.Tab(
                            label="Treemap",
                            children=[
                                dcc.Loading(
                                    id="loading-treemap",
                                    children=[dcc.Graph(id="topic-treemap")],
                                    type="default",
                                )
                            ],
                        ),
                    ],
                )
            ],
            md=8,
        ),
    ],
    className="g-0"
)


TOPIC_INFO_GRAPHS = dbc.Row(
    [
        dbc.Col(
            dcc.Loading(
                id="loading-topic-years",
                children=[
                    dcc.Graph(id="topic-years-histogram")
                ],
                type="default",
            )
        ),
        dbc.Col(
            dcc.Loading(
                id="loading-topic-courts",
                children=[
                    dcc.Graph(id="topic-courts-graph")
                ],
                type="default",
            ),
            md=4
        )
    ], className="g-0"
)


TOPIC_CARD = [
    dbc.CardHeader(id="topic-header", children=[html.H5("Select a topic", id="selected_topic_name")]),
    dbc.Alert(
        "Not enough data to render these plots, please adjust the filters",
        id="no-topic-data-alert",
        color="warning",
        style={"display": "none"},
    ),
    dbc.CardBody(
        [TOPIC_WORDS_GRAPHS, TOPIC_INFO_GRAPHS]
    )
]


# ##### Body

BODY = dbc.Container(
    [
        SEARCH_BOX,
        dbc.Card(WORD_CARD, style={"marginTop": 20}),
        dbc.Card(TOPIC_CARD, style={"marginTop": 20, "marginBottom": 30}),
    ],
    className="mt-12",
)


# ##### Callbacks


@app.callback(
    [
        Output("words-drop", "options"),
        Output("words-drop", "value"),
    ],
    Input('search-button', 'n_clicks'),
    State('search-input', 'value')
)
def populate_search_dropdown(n_clicks, searches):
    if not searches:
        return [], None
    options = []
    for search in searches.split("-"):
        search.strip()
        options.append({"label": search, "value": search})
    return options, options[0]['value']


@app.callback(
    Output("similar-context-graph", "figure"),
    Input("words-drop", "value")
)
def get_similar_context_graph(search):
    if not search:
        return {}
    words = [word.strip() for word in search.split(",")]
    sim = app_loader.get_n_similar(word=words, n=15, model_type="full")[::-1]
    if not sim:
        return {}
    return px.histogram(
        y=[word[0] for word in sim],
        x=[word[1] for word in sim],
        orientation="h",     
        title="Similar context",
        color_discrete_sequence=['darkturquoise']
    ).update_layout(
        template=TEMPLATE,
        xaxis_title='',
        yaxis_title=''
    )


@app.callback(
    Output('grams-graph', 'figure'),
    [
        Input('search-button', 'n_clicks'),
        Input("semantic-year-slider", "value")
    ],
    [State('search-input', 'value')])
def update_output(n_clicks, year, searches):
    if not searches:
        return {}
    
    searches = searches.split("-")
    fig = go.Figure(layout=go.Layout(
        title="Semantic shift - yearly",
        template=TEMPLATE,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(count=5,
                         label="5y",
                         step="year",
                         stepmode="backward"),
                    dict(count=10,
                         label="10y",
                         step="year",
                         stepmode="backward"),
                    dict(count=25,
                         label="25y",
                         step="year",
                         stepmode="backward"),
                    dict(count=50,
                         label="50y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            title="year",
            type="date"
        ),
        yaxis=dict(title="freq")
    ))
    for search in searches:
        words = [word.strip() for word in search.split(",")]
        grams = app_loader.get_freq_distribution(words, interval=10)
        if not grams:
            continue
        fig.add_trace(go.Scatter(x=[year_perc[0] for year_perc in grams], y=[year_perc[1] for year_perc in grams],
                            mode='lines',
                            name=search))
    return fig


@app.callback(
    Output('semantic-epoch-shift-graph', 'figure'),
    [
        Input('search-button', 'n_clicks'),
    ],
    [State('search-input', 'value')])
def update_output(n_clicks, searches):
    if not searches:
        return {}
    
    searches = searches.split("-")
    fig = go.Figure(layout=go.Layout(
        title="Semantic shift - epoch",
        template=TEMPLATE,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(count=5,
                         label="5y",
                         step="year",
                         stepmode="backward"),
                    dict(count=10,
                         label="10y",
                         step="year",
                         stepmode="backward"),
                    dict(count=25,
                         label="25y",
                         step="year",
                         stepmode="backward"),
                    dict(count=50,
                         label="50y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            title="year",
            type="date"
        ),
        yaxis=dict(title="freq")
    ))
    for search in searches:
        search = search.strip()
        words = [word.strip() for word in search.split(",")]
        semantic_shift = [sem for sem in app_loader.get_semantic_data(words)['ten_year'] if sem[1] != -1]
        if not semantic_shift:
            continue
        fig.add_trace(go.Scatter(
            x=[year_perc[0] for year_perc in semantic_shift], y=[year_perc[1] for year_perc in semantic_shift],
            mode='lines',
            name=search,
            customdata=[search]*len(semantic_shift)
        ))
    return fig


@app.callback(
    [
        Output('semantic-epoch-first-graph', 'figure'),
        Output('semantic-epoch-second-graph', 'figure'),
    ],
    [
        Input('semantic-epoch-shift-graph', 'clickData'),
        Input('search-button', 'n_clicks'),
    ],
    State('search-input', 'value')
)
def compare_epoch_semantic_shift(clickData, n_clicks, searches):
    ctx = dash.callback_context 
    if ctx.triggered[0]['prop_id'].split('.')[0] == "search-button" or not clickData or not len(clickData['points']) > 0:
        return {}, {}
    click = clickData['points'][0]
    words = click['customdata']
    year = int(click['x'].split('-')[0])

    context_preceding = app_loader.get_n_similar(word=[word.strip() for word in words.split(",")], n=15, model_type="ten", year=year - 10)[::-1]
    context_current = app_loader.get_n_similar(word=[word.strip() for word in words.split(",")], n=15, model_type="ten", year=year)[::-1]
    
    return px.histogram(
        y=[word[0] for word in context_preceding],
        x=[word[1] for word in context_preceding],
        orientation="h",     
        title=f"{words} - {year - 10}",
        color_discrete_sequence=['darkturquoise']
    ).update_layout(
        template=TEMPLATE,
        xaxis_title='',
        yaxis_title=''
    ) if context_preceding else {}, px.histogram(
        y=[word[0] for word in context_current],
        x=[word[1] for word in context_current],
        orientation="h",     
        title=f"{words} - {year}",
        color_discrete_sequence=['darkturquoise']
    ).update_layout(
        template=TEMPLATE,
        xaxis_title='',
        yaxis_title=''
    ) if context_current else {}


@app.callback(
    [
        Output('semantic-yearly-shift-graph', 'figure'),
        Output("semantic-year-slider", "marks"),
        Output("semantic-year-slider", "min"),
        Output("semantic-year-slider", "max"),
        Output("semantic-year-slider", "value"),
    ],
    [
        Input('search-button', 'n_clicks'),
        Input("semantic-year-slider", "value")
    ],
    [State('search-input', 'value')])
def update_output(n_clicks, year, searches):
    if not searches:
        return {}, {}, 0, 0, 0
    
    searches = searches.split("-")
    fig = go.Figure(layout=go.Layout(
        title="Semantic shift - epoch",
        template=TEMPLATE,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(count=5,
                         label="5y",
                         step="year",
                         stepmode="backward"),
                    dict(count=10,
                         label="10y",
                         step="year",
                         stepmode="backward"),
                    dict(count=25,
                         label="25y",
                         step="year",
                         stepmode="backward"),
                    dict(count=50,
                         label="50y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            title="year",
            type="date"
        ),
        yaxis=dict(title="freq")
    ))
    min_year = None
    max_year = None
    for search in searches:
        search = search.strip()
        words = [word.strip() for word in search.split(",")]
        semantic_shift = [sem for sem in app_loader.get_semantic_data(words, base_year=year if year != 0 else 2010)['one_year'] if sem[1] != -1]
        if not semantic_shift:
            continue
        min_year = min(min_year, semantic_shift[-1][0]) if min_year else semantic_shift[-1][0] 
        max_year = max(max_year, semantic_shift[0][0]) if max_year else semantic_shift[0][0]
        fig.add_trace(
            go.Scatter(
                x=[year_perc[0] for year_perc in semantic_shift], y=[year_perc[1] for year_perc in semantic_shift],
                mode='lines',
                name=search,
                customdata=[search]*len(semantic_shift)
            )
        )
    if not min_year:
        min_year = 1770
    if not max_year:
        max_year = 2010
    return fig, {min_year: f"{min_year}", max_year: f"{max_year}"}, min_year, max_year, year if min_year <= year <= max_year else max_year


@app.callback(
    [
        Output('semantic-yearly-first-graph', 'figure'),
        Output('semantic-yearly-second-graph', 'figure'),
    ],
    [
        Input('semantic-yearly-shift-graph', 'clickData'),
        Input("semantic-year-slider", "value"),    
        Input('search-button', 'n_clicks'),
    ],
    State('search-input', 'value')
)
def compare_epoch_semantic_shift(clickData, selected_year, n_clicks, searches):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'].split('.')[0] == "search-button" or not clickData or not len(clickData['points']) > 0:
        return {}, {}
    click = clickData['points'][0]
    words = click['customdata']
    clicked_year = int(click['x'].split('-')[0])

    context_selected = app_loader.get_n_similar(word=[word.strip() for word in words.split(",")], n=15, model_type="one", year=selected_year)[::-1]
    context_clicked = app_loader.get_n_similar(word=[word.strip() for word in words.split(",")], n=15, model_type="one", year=clicked_year)[::-1]
    
    return px.histogram(
        y=[word[0] for word in context_selected],
        x=[word[1] for word in context_selected],
        orientation="h",     
        title=f"{words} - {selected_year}",
        color_discrete_sequence=['darkturquoise']
    ).update_layout(
        template=TEMPLATE,
        xaxis_title='',
        yaxis_title=''
    ) if context_selected else {}, px.histogram(
        y=[word[0] for word in context_clicked],
        x=[word[1] for word in context_clicked],
        orientation="h",     
        title=f"{words} - {clicked_year}",
        color_discrete_sequence=['darkturquoise']
    ).update_layout(
        template=TEMPLATE,
        xaxis_title='',
        yaxis_title=''
    ) if context_clicked else {}


@app.callback(
    [
        Output("word-topics-graph", "figure"),
        Output("topic-descriptions-list", "children"),
    ],
    [
        Input('search-button', 'n_clicks'),
        Input('word-topics-tabs', 'value')
    ],
    State('search-input', 'value'))
def get_generic_topics_radar_graph(n_clicks, tab, searches):
    if not searches:
        return {}, []

    fig = go.Figure(layout=go.Layout(
            title="Topic distribution",
            template=TEMPLATE,
    )               )
    
    searches = searches.split("-")
    for search in searches:
        words = [word.strip() for word in search.split(",")]
        
        topics = app_loader.get_topic_dist(words, model="big" if tab == "General" else "small")
        values = numpy.array(list(topics.values()), dtype='f') * 100 / max(topics.values())
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[str(name) for name in list(topics.keys())],
            fill='toself',
            name=search,
            hoverinfo="text",
            textposition="top center",       
            hovertext=[f"Topic {topic} - {format(value, '.2f')}%" for topic, value in zip(topics, values)]
        ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=False,
          range=[0, 100]
        )),
      showlegend=True
    )
    
    list_topics =[
            dbc.ListGroupItem(
                html.Div(
                    [
                        html.H6(f"Topic {i}", className="mb-1"),
                        html.Small(f"{app_loader.get_topics_description(i, category=tab)}", className="text-muted"),
                    ], 
                    className="d-flex w-100 justify-content-between",
                ), 
                id={"type": "topic-button", "index": i},
                action=True,
                n_clicks=0,
            )
            for i in range(0, len(topics))
        ]
    
    return fig, list_topics


def get_wordcloud_graphs_topic_words(word_cloud):
    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []

    for (word, freq), fontsize, position, orientation, color in word_cloud.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # get the positions
    x_arr = []
    y_arr = []
    for i in position_list:
        x_arr.append(i[0])
        y_arr.append(i[1])

    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i * 80)

    trace = go.Scatter(
        x=x_arr,
        y=y_arr,
        textfont=dict(size=new_freq_list, color=color_list),
        hoverinfo="text",
        textposition="top center",
        hovertext=["{0} - {1}".format(w, f) for w, f in zip(word_list, freq_list)],
        mode="text",
        text=word_list,
    )

    layout = go.Layout(
        {
            "xaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "range": [-100, 250],
            },
            "yaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "range": [-100, 450],
            },
            "margin": dict(t=20, b=20, l=10, r=10, pad=4),
            "hovermode": "closest",
        }
    )

    return {"data": [trace], "layout": layout}


@app.callback(
    [
        Output("topic-header", "children"),
        Output("topic-top-words-graph", "figure"),
        Output("topic-treemap", "figure"),
        Output("topic-wordcloud", "figure")
    ],
    [
        Input('word-topics-graph', 'clickData'),
        Input('word-topics-tabs', 'value'),
        Input({'type': 'topic-button', 'index': ALL}, 'n_clicks'),
    ]
)
def get_topic_words_radar_graph(selected_topic, tab, n_click):
    context = dash.callback_context.triggered[0]
    is_button = False
    try:
        if ast.literal_eval(context['prop_id'].split(".")[0])['type'] == "topic-button":
            is_button = True
    except:
        pass

    if not is_button and selected_topic:
        topic_id = selected_topic['points'][0]['pointNumber']
    elif is_button:
        topic_id = ast.literal_eval(context['prop_id'].split(".")[0])['index']
    else:
        return [html.H5(f"Select a topic")], {}, {}, {}
    
    sim = app_loader.get_topics_words(n=80, model="big" if tab == "General" else "small")[topic_id]
    
    if not sim:
        return [html.H5("Select a topic")], {}, {}, {}
    
    max_freq = sim[0][1]
    words = [word[0] for word in sim]
    freqs = [word[1]/max_freq for word in sim]
    
    treemap_trace = go.Treemap(
        labels=words[:40], parents=[""] * len(words[:40]), values=freqs
    )
    treemap_layout = go.Layout({"margin": dict(t=0, b=0, l=0, r=0, pad=0)})
    treemap_figure = {"data": [treemap_trace], "layout": treemap_layout}
    
    wc = WordCloud().generate_from_frequencies(frequencies={word[0]: word[1] for word in sim})
    wordcloud = get_wordcloud_graphs_topic_words(wc)
    
    return [html.H5(f"{tab} Topic {topic_id} - {app_loader.get_topics_description(topic_id, category=tab)}")],px.histogram(
        y=words[:20][::-1],
        x=freqs[:20][::-1],
        orientation="h",     
        color_discrete_sequence=['darkturquoise']
    ).update_layout(
        template=TEMPLATE,
        xaxis_title='',
        yaxis_title='',
        height=550
    ), treemap_figure, wordcloud


@app.callback(
    Output("topic-years-histogram", "figure"),
    [
        Input('word-topics-graph', 'clickData'),
        Input('word-topics-tabs', 'value'),
        Input({'type': 'topic-button', 'index': ALL}, 'n_clicks'),
    ],)
def get_topic_years_histogram(selected_topic, tab, n_clicks):
    
    context = dash.callback_context.triggered[0]
    is_button = False
    try:
        if ast.literal_eval(context['prop_id'].split(".")[0])['type'] == "topic-button":
            is_button = True
    except:
        pass

    if not is_button and selected_topic:
        topic_id = selected_topic['points'][0]['pointNumber']
    elif is_button:
        topic_id = ast.literal_eval(context['prop_id'].split(".")[0])['index']
    else:
        return {}
  
    topic_dists = app_loader.get_topics_date_distribution(interval=5)[topic_id] 
    years = [year_freq[0] for year_freq in topic_dists]
    freqs = [year_freq[1] for year_freq in topic_dists]
    data = [
        {
            "x": years,
            "y": freqs,
            "text": years,
            "type": "bar",
            "name": "",
        }
    ]
    layout = {
        "autosize": True,
        "margin": dict(t=10, b=20, l=40, r=0, pad=4),
        "xaxis": dict(rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(count=5,
                     label="5y",
                     step="year",
                     stepmode="backward"),
                dict(count=10,
                     label="10y",
                     step="year",
                     stepmode="backward"),
                dict(count=25,
                     label="25y",
                     step="year",
                     stepmode="backward"),
                dict(count=50,
                     label="50y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
                       rangeslider=dict(
                            visible=True
                        ),
                       title="year",
                       type="date",
                       showticklabels=True, )
    }
    return {"data": data, "layout": layout}


@app.callback(
    Output("topic-courts-graph", "figure"),
    [
        Input('word-topics-graph', 'clickData'),
        Input('word-topics-tabs', 'value'),
        Input({'type': 'topic-button', 'index': ALL}, 'n_clicks'),
    ],)
def get_topic_courts_distribution_radar_graph(selected_topic, tab, n_clicks):
    context = dash.callback_context.triggered[0]
    is_button = False
    try:
        if ast.literal_eval(context['prop_id'].split(".")[0])['type'] == "topic-button":
            is_button = True
    except:
        pass

    if not is_button and selected_topic:
        topic_id = selected_topic['points'][0]['pointNumber']
    elif is_button:
        topic_id = ast.literal_eval(context['prop_id'].split(".")[0])['index']
    else:
        return {}  

    court_freqs = app_loader.get_topics_court_distribution(model="big" if tab == "General" else "small")[topic_id]

    fig = go.Figure(
        layout=go.Layout(
            title="Courts distribution",
            template=TEMPLATE,
        )               
    )
    
    values = [freq * 100 for freq in court_freqs.values()]
    max_value = max(values) if values else 1
        
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=[court.split("Illinois")[1].strip() for court in list(court_freqs.keys())],
        fill='toself',
        name=str(topic_id),
        hoverinfo="text",
        textposition="top center",       
        hovertext=[f"{round(value, 2)}%" for value in values],
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=False,
          range=[0, int(max_value * 1.5) if max_value < 50 else int(max_value + 10)]
        )
      )
    )
    return fig

app.layout = html.Div(children=[NAVBAR, BODY])


def _terminate_server_for_port(host, port):
    shutdown_url = "http://{host}:{port}/_shutdown_{token}".format(
        host=host, port=port, token=Dash._token
    )
    try:
        response = requests.get(shutdown_url)
    except Exception as e:
        pass


if __name__ == '__main__':
    app.run_server(dev_tools_ui=True, debug=True,
                   dev_tools_hot_reload=True, threaded=True)

