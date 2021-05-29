from models.ClassificadorTweet import ClassificadorTweet
from models.ClassificadorNews import ClassificadorNews

if __name__ == '__main__':
    clf = ClassificadorTweet()
    print(clf.predict('Economia Meirelles comemora fim da recessão mas diz que retomada não é uma linha reta'))

    clf = ClassificadorNews()
    print(clf.predict("​O amigo de Michel Temer, José Yunes, disse que atuou como “mula involuntária” do ministro Eliseu Padilha ao receber um pacote no seu escritório de advocacia, das mãos do doleiro Lúcio Funaro. Mas o que mais chocou o país não foi a denúncia em si, mas a revelação de que Temer tem um amigo. “Não noticiaram nem como aliado, nem como subordinado, mas como amigo do Temer. Isso surpreendeu a todos da minha família aqui em casa”, disse Rebeca Sampaio, brasileira de 36 anos. O ex-presidente Lula se queixou, em entrevista coletiva, dizendo que até escândalos de corrupção envolvendo amigos pessoais Temer tomou dele. “Esse golpista quer tomar tudo que é meu, da Dilma e do PT”, disse o petista."))