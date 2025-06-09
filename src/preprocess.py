import re
import spacy

# Carregar modelo em português
nlp = spacy.load("pt_core_news_lg")

# Função para corrigir abreviações comuns
def abbreviation_correction(text):
    # Dicionário de abreviações comuns
    abbreviations = {
        'abast': 'abastecimento',
        'abastec': 'abastecimento',
        'abert': 'abertura',
        'alim': 'alimentacao',
        'alinhmto': 'alinhamento',
        'anor': 'anormal',
        'automat': 'automatico',
        'aux': 'auxiliar',
        'bba': 'bomba',
        'bbas': 'bomba',
        'bc': 'braco carregamento',
        'benz': 'benzeno',
        'bio': 'biologico',
        'biol': 'biologico',
        'biolog': 'biologico',
        'biologic': 'biologico',
        'biolo': 'biologico',
        'bloq': 'bloqueio',
        'bomb': 'bomba',
        'borb': 'borboleta',
        'borbol': 'borboleta',
        'capac': 'capacitor',
        'cent': 'centrifuga',
        'centr': 'centrifuga',
        'centrif': 'centrifuga',
        'cmd': 'comando',
        'coman': 'comando',
        'combat': 'combate',
        'combust': 'combustao',
        'conj': 'conjunto',
        'contr': 'controle',
        'control': 'controle',
        'controlad': 'controladora',
        'contralavag': 'contralavagem',
        'deriv': 'derivacao',
        'desc': 'descarga',
        'desidrat': 'desidratado',
        'desidratad': 'desidratado',
        'despej': 'despejo',
        'difer': 'diferencial',
        'disj': 'disjuntor',
        'dosad': 'dosador',
        'dren': 'drenagem',
        'el': '',
        'elem': 'elemento',
        'elet': 'eletrico',
        'eletr': 'eletrico',
        'emer': 'emergencia',
        'entr': 'entrada',
        'env': 'envio',
        'eq': 'equalizacao',
        'equaliz': 'equalizacao',
        'esfer': 'esfera',
        'esgot': 'esgotamento',
        'exc': 'excentrica',
        'fech': 'fechamento',
        'filt': 'filtro',
        'filtr': 'filtro',
        'floculac': 'floculacao',
        'fluid': 'fluido',
        'gav': 'gaveta',
        'glb': 'gleba',
        'hor': 'horizontal',
        'horiz': 'horizontal',
        'ilum': 'iluminacao',
        'inc': 'incendio',
        'ind': 'indicador',
        'indic': 'indicador',
        'indicad': 'indicador',
        'indus': 'industrial',
        'inor': 'inorganico',
        'inorg': 'inorganico',
        'inorga': 'inorganico',
        'int': 'interna',
        'inter': 'interna',
        'interl': 'interligacao',
        'intern': 'interna',
        'lab': 'laboratorio',
        'laborat': 'laboratorio',
        'linh': 'linha',
        'lub': 'lubrificacao',
        'lubri': 'lubrificacao',
        'lubrif': 'lubrificacao',
        'manif': 'manifold',
        'mec': 'mecanico',
        'merej': 'merejamento',
        'mnf': 'manifold',
        'mot': 'motor',
        'motoriz': 'motorizada',
        'mov': 'movimento',
        'ofic': 'oficina',
        'permutad': 'permutador',
        'platafor': 'plataforma',
        'platafororma': 'plataforma',
        'plc': 'clp',
        'pn': 'painel',
        'polieletro': 'polieletrolito',
        'polieletrol': 'polieletrolito',
        'pot': 'potencia',
        'preboc': 'rebocador',
        'pres': 'pressao',
        'press': 'pressurizacao',
        'pressur': 'pressurizacao',
        'prim': 'primario',
        'primar': 'primario',
        'prin': 'principal',
        'princ': 'principal',
        'refr': 'refrigeracao',
        'refriger': 'refrigeracao',
        'remo': 'remoto',
        'resfr': 'resfriamento',
        'ret': 'retencao',
        'retenc': 'retencao',
        'secag': 'secagem',
        'selag': 'selagem',
        'sinaliz': 'sinalizacao',
        'sist': 'sistema',
        'succ': 'succao',
        'st': 'sump tanque',
        'sumptank': 'sump tanque',
        'temp': 'temperatura',
        'temperat': 'temperatura',
        'tank': 'tanque',
        'tq': 'tanque',
        'trans': 'transmissor',
        'transf': 'transferencia',
        'transi': 'transicao',
        'transm': 'transmissor',
        'val': 'valvula',
        'valv': 'valvula',
        'valvs': 'valvula',
        'vert': 'vertical',
        'vaz': 'vazamento',
        'vazm': 'vazamento',
        'vazmento': 'vazamento',
        'vazmto': 'vazamento',
        'vzmto': 'vazamento',
        'vzto': 'vazamento',
        'ventilad': 'ventilador'
    }
    
    # Substitui as abreviações no texto
    for abbr, full in abbreviations.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text)
    
    return text
    
def preprocess_text(text):
    # Converte para minúsculas
    text = text.lower()
    # Substitui acentos por caracteres sem acento
    text = re.sub(r'[áàãâä]', 'a', text)
    text = re.sub(r'[éèêë]', 'e', text)
    text = re.sub(r'[íìîï]', 'i', text)
    text = re.sub(r'[óòôõö]', 'o', text)
    text = re.sub(r'[úùûü]', 'u', text)
    text = re.sub(r'[ç]', 'c', text)
    # Remove hífens dos tags de equipamentos
    #text = re.sub(r'([a-z]*)\-([0-9]*)', r'\1\2', text)
    # Remove caracteres especiais e números
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove espaços extras
    text = ' '.join(text.split())
    # Corrige abreviações comuns
    text = abbreviation_correction(text)
    # Lematiza o texto
    doc = nlp(text)
    text = ' '.join([token.lemma_ for token in doc if not token.is_stop and len(token.text) > 1])
    if text.__contains__('valvular'):
        text = text.replace('valvular', 'valvula')
    return text
