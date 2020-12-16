import sys
sys.path.append("..")
import unittest
import json
import os
from feature_extraction import FeatureExtraction
from embedding.model import EmbeddingModel
import numpy as np

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.embedder = EmbeddingModel()
        self.fe = FeatureExtraction(self.embedder)
        self.articles = []
        with open("test/jsonl/test_articles.jsonl", "r") as f:
            for line in f:
                article = json.loads(line)
                self.articles.append(article)

    def test_get_first_paragraph(self):
        actual_text = []
        expected_text = [
            "Die Stadt Karlsruhe will das „Sicherheitsgefühl“ in der Innenstadt verbessern. Die rechtlichen Voraussetzungen für eine polizeiliche Videoüberwachung sind nicht erfüllt. Eine Entwicklung des Energiekonzerns EnBW könnte diese Hürde umgehen.",
            "Der zweite Tag des 32. Chaos Communication Congress ist noch nicht ganz vorüber, aber wir werfen schon mal einen Blick auf die bisherigen Presseberichte und haben ein paar Empfehlungen für den morgigen, dritten Tag zusammengestellt.",
            "Teil I: Es gibt viele Gründe, zum Chaos Communication Congress zu fahren. Ebenso viele sprechen dagegen. Vermutlich stimmt beides. Ein Erlebnisbericht von ebenjenem.",
            "Teil II Was ist das? Was sagt er da? Was machen die da? Der Congress-Neuling schnappt Dinge auf, hört zu und beobachtet. Es ergibt sich ein Text, der nicht all zu leicht zu verstehen ist."
        ]
        for article in self.articles:
            first_paragraph = self.fe.get_first_paragraph(article)
            actual_text.append(first_paragraph)
        
        for id, actual in enumerate(actual_text):
            self.assertEqual(actual, expected_text[id])

    def test_get_first_paragraph_with_titles(self):
        actual_text = []
        expected_text = [
            "Karlsruhe Energieversorger plant automatisierte Videoüberwachung Die Stadt Karlsruhe will das „Sicherheitsgefühl“ in der Innenstadt verbessern. Die rechtlichen Voraussetzungen für eine polizeiliche Videoüberwachung sind nicht erfüllt. Eine Entwicklung des Energiekonzerns EnBW könnte diese Hürde umgehen.",
            "#netzrückblick: 32C3 – der zweite Tag Der zweite Tag des 32. Chaos Communication Congress ist noch nicht ganz vorüber, aber wir werfen schon mal einen Blick auf die bisherigen Presseberichte und haben ein paar Empfehlungen für den morgigen, dritten Tag zusammengestellt.",
            "Entscheide dich und bleib dabei. Erlebnisbericht eines Neulings auf dem Chaos Communication Congress Teil I: Es gibt viele Gründe, zum Chaos Communication Congress zu fahren. Ebenso viele sprechen dagegen. Vermutlich stimmt beides. Ein Erlebnisbericht von ebenjenem.",
            "Ein Text, den ich noch nicht lesen kann. Erlebnisbericht eines Neulings auf dem Chaos Communication Congress Teil II Was ist das? Was sagt er da? Was machen die da? Der Congress-Neuling schnappt Dinge auf, hört zu und beobachtet. Es ergibt sich ein Text, der nicht all zu leicht zu verstehen ist."
        ]
        for article in self.articles:
            q_text = self.fe.get_first_paragraph_with_titles(article)
            actual_text.append(q_text)
        for id, actual in enumerate(actual_text):
            self.assertEqual(actual, expected_text[id])

    def test_get_line_separated_text_tokens(self):
        actual_tokens = self.fe.get_line_separated_text_tokens(self.articles[0])
        expected_tokens = [
            "Die Stadt Karlsruhe will das „Sicherheitsgefühl“ in der Innenstadt verbessern. Die rechtlichen Voraussetzungen für eine polizeiliche Videoüberwachung sind nicht erfüllt. Eine Entwicklung des Energiekonzerns EnBW könnte diese Hürde umgehen.",
            "Der Europaplatz ist ein belebter Umsteigepunkt der Karlsruher Innenstadt. Viele Menschen kommen hier täglich vorbei, ob beim Pendeln, zum Einkaufen oder um ins Kino oder Restaurant zu gehen. Obwohl es keine auffällig hohe Kriminalität gibt, will der Energiekonzern EnBW gemeinsam mit der Stadtverwaltung eine besondere Form der Videoüberwachung installieren. Da eine polizeiliche Überwachung wegen der niedrigen Kriminalität nicht zulässig wäre, setzt EnBW auf ein System, das mit einer automatischen Verfremdung der aufgezeichneten Personen als „datenschutzkonforme Videoüberwachung“ dargestellt wird.",
            "EnBW erläuterte das System namens SAVAS DS+ gegenüber netzpolitik.org als „so konzipiert, dass keine personenbezogenen Daten von den Sensoren aufgenommen werden“, da einerseits mit niedriger Punktdichte aufgezeichnet und die Bilddaten zusätzlich automatisch in „eine Art Schattendarstellung“ umgewandelt würden. Die Auswertung der Daten soll durch eine „künstliche Intelligenz“ erfolgen, die die Situation am Europaplatz analysiert: „Wie viele Personen sind vor Ort? Stehen diese in Gruppen oder bewegen sich diese? Mittelfristig ist auch die Analyse von bestimmten Verhaltensmustern angedacht, um zum Beispiel Schlägereien schneller erkennen zu können.“",
            "Während sogenannte smarte Überwachung, die Menschen oder „auffälliges Verhalten“ automatisch identifizieren soll, bisher häufig durch die Datenschutzgrundverordnung eingeschränkt wird, könnte EnBW hier ein System etablieren, das durch die Verfremdung kaum gesetzlichen Hürden begegnet.",
            "Private Videoüberwachung für die Polizei",
            "Um die Erkennung bestimmter Personen soll es dabei aber erst einmal nicht gehen. Zunächst wären vorrangig Mitarbeiter von EnBW im Einsatz, die auf Basis der Informationen aus dem System die Lage einschätzen und „den direkten Kontakt zur Polizei“ halten.",
            "Ziel des Projekts ist der Aufbau einer Überwachungsinfrastruktur zur Unterstützung der Polizei, wie auch EnBW bestätigt: „Das System ist gedacht, um Polizeieinsätze künftig schneller und effektiver durchführen zu können.“ Noch 2018 betonte Caren Denner, Polizeipräsidentin Karlsruhes, dass es keine Videoüberwachung einzelner Orte der Stadt geben würde, da die rechtlichen Voraussetzungen für einen Kriminalitätshotspot nirgends gegeben sein.",
            "Inwieweit diese Form der Überwachung zulässig ist, ist also zumindest fragwürdig: Umgeht diese Konstellation durch die Vergabe an EnBW nicht einfach polizeirechtliche Einschränkungen? Mitarbeiter der in Landeshand befindlichen Aktiengesellschaft kommen hier für quasi-polizeiliche Aufgaben zum Einsatz. Ähnliche Entwicklungen hat Arne Semsrott auf Bundesebene im Bezug auf die Informationsfreiheit treffend beschrieben: Um eine Kontrolle durch die Öffentlichkeit zu erschweren, erfolgt auch hier eine Auslagerung staatlicher Aufgaben an Unternehmen als eine „Flucht ins Privatrecht“.",
            "EnBW hingegen vergleicht das System lediglich „mit einem Anwohner des Europaplatzes, der zum Beispiel bei einer Schlägerei in der Nacht die Polizei anruft“ – eine fragwürdige Analogie für ein aufwändiges System mit Kameras, automatischer Datenauswertung und mehreren Mitarbeitern in einer Alarmempfangsstelle.",
            "Die Stadtverwaltung und EnBW beharren darauf, dass keine rechtlichen Einschränkungen des Projekts bestünden, weil durch die niedrige Auflösung und die Verfremdung des Bildes „zu keinem Zeitpunkt Rückschlüsse auf konkrete Personen gezogen werden“ können.",
            "Auch die Behörde des Landesdatenschutzbeauftragten von Baden-Württemberg gab gegenüber netzpolitik.org an, dass keine Bedenken bestehen. Den Schilderungen von EnBW folgend sei das System datenschutzrechtlich nicht zu beanstanden. Das System selbst hat die Datenschutzbehörde jedoch noch nicht geprüft, da das die „personellen Möglichkeiten der Dienststelle“ übersteige.",
            "Gesetzliche Regelung und mehr Transparenz nötig",
            "Mit dem System von EnBW könnte sich hier eine Form der Videoüberwachung etablieren, die von der aktuellen Gesetzeslage nicht erfasst und kaum Einschränkungen ausgesetzt ist. Doch selbst wenn sich konkrete Personen tatsächlich nicht identifizieren lassen, hat die Videoüberwachung negative Folgen für uns alle: Sämtliche Passant:innen des belebten Platzes werden unter Generalverdacht gestellt. Menschen, die sich überwacht fühlen, verhalten sich anders und neigen dazu, sich nicht mehr frei auszudrücken.",
            "Insbesondere für zentrale Orte des öffentlichen Lebens wie den Europaplatz ist das gefährlich: Er stellt als Tor zur innerstädtischen Fußgängerzone einen zentralen Ort des Karlsruher Nacht- und Alltagsleben dar und ist auch häufig Bühne für politische Kundgebungen und Demonstrationen.",
            "Unklar für die Passant:innen ist ebenfalls, welche Verhaltensweisen und Bewegungsmuster als verdächtig eingestuft werden. Unter Umständen könnten also schon längere Aufenthalte an einem Ort oder größere Menschengruppen zu Verdachtsmomenten und polizeilichen Kontrollen führen. Audioaufnahmen würden laut EnBW nicht angefertigt, allerdings würden „Lautstärkelevel und Frequenzverteilungen ermittelt“. Gespräche können also nicht aufgezeichnet werden, dafür könnten lautes Lachen oder Verkehrsgeräusche dem Algorithmus bereits verdächtig erscheinen.",
            "Evaluierung des Projekts noch nicht festgelegt",
            "Ein Vertreter der Stadt sagte gegenüber netzpolitik.org, dass „die konkrete Festlegung, welche Kriterien für eine Evaluation herangezogen werden, noch nicht erfolgt [ist]“. Somit ist ungewiss, nach welchen Maßstäben das Projekt letztlich als Erfolg oder Misserfolg bewertet werden würde. Eine ähnliche Beliebigkeit in der Evaluierung hatte schon am Berliner Überwachungsbahnhof Südkreuz dazu geführt, dass ein Überwachungsprojekt trotz fragwürdiger Erkennungsquoten als Erfolg dargestellt wurde.",
            "Da der Europaplatz der einzige Platz Karlsruhes wäre, an dem Tag und Nacht Videodaten von EnBW-Mitarbeitern ausgewertet würden, wäre ein Anstieg von Polizeieinsätzen wenig überraschend. Dieser könnte wiederum zur Rechtfertigung weiterer Maßnahmen herangezogen werden, auch wenn sich am eigentlichen Kriminalitätsgeschehen faktisch nichts geändert hätte.",
            "Im konkreten Karlsruher Fall ist neben den rechtlichen Fragen zudem unklar, welchen Mehrwert eine Videoüberwachung des Europaplatzes überhaupt hätte. Eine 2018 vom Heidelberger Kriminologen Prof. Dieter Hermann durchgeführte Studie zu Karlsruhe attestierte der Stadt eine gute Situation in Bezug auf Kriminalität und Kriminalitätsfurcht. Gegenüber den Vergleichsstädten Mannheim und Heidelberg befinde sich „die Kriminalitätsfurcht in Karlsruhe auf einem vergleichsweise niedrigen Niveau“, die Befragungsergebnisse sprechen nur für „einen leichten Anstieg der Kriminalitätsfurcht in Karlsruhe“. Anstelle von Kriminalität nannten die Befragten insgesamt eher rücksichtslose Autofahrer:innen, rechtswidriges Parken und Schmutz und Müll als vorherrschende Probleme.",
            "Präventive und bauliche Maßnahmen statt Videoüberwachung",
            "Videoüberwachung erwähnte die Studie übrigens mit keinem Wort als probates Mittel zur Senkung von Kriminalität oder Steigerung des Sicherheitsgefühls. Insbesondere mit Blick auf den Europaplatz und alkoholisierte Jugendgruppen werden Sozialarbeiter:innen und vorbeugende Aufklärungsmaßnahmen empfohlen. Vor allem die Karlsruher CDU-Fraktion setzt jedoch weiterhin auf den Ausbau der Überwachungsmaßnahmen durch dieses Pilotprojekt, nachdem sie 2018 mit ihrer Forderung nach einer Videoüberwachung nach Mannheimer Modell gescheitert war.",
            "Es ist absehbar, dass eine Videoüberwachung des Platzes zur Abschreckung alkoholisierter Gruppen wenig effektiv wäre. Deutlich zielführender zur Steigerung des „subjektiven Sicherheitsgefühls“ am Karlsruher Europaplatz wären wohl eine Verbesserung der Verkehrssituation oder bauliche Umgestaltungen des Platzes – solche Maßnahmen tauchen auch als Vorschläge aus der Bevölkerung im Sicherheitskonzept der Stadt auf. Dies würde auch insgesamt zu einer Aufwertung des Platzes führen. Zwar wird der Europaplatz täglich von vielen Menschen passiert, doch wenige verbringen hier gern Zeit.",
            "Auch die Karlsruher Stadträtin Mathilde Göttel (DIE LINKE) äußert sich kritisch: „Die geplante Videoüberwachung muss man als bloßen Aktionismus verstehen. Mit einem Umbau des Platzes könnte man das Sicherheitsgefühl viel nachhaltiger verbessern und dabei auch noch einen Ort mit Aufenthaltsqualität gewinnen.“ Im Gegensatz zur geplanten Videoüberwachung, die EnBW für die dreijährige Versuchsphase kostenlos anbieten will, würde das allerdings Geld kosten. Voraussichtlich nächsten Monat soll der mehrheitlich grüne Gemeinderat unter Oberbürgermeister Frank Mentrup über das Projekt abstimmen."
        ]
        for id, actual in enumerate(actual_tokens):
            self.assertEqual(actual, expected_tokens[id])

    def test_mean_of_pairwise_cosine_distances(self):
        ems = np.array([
            [-1,1,1],
            [-11,3,9],
            [22,0,8]
        ], dtype=float)
        self.assertTrue(abs(0.9770-self.fe.mean_of_pairwise_cosine_distances(ems)) < 1e-4)

    def test_mean_of_pairwise_cosine_distances_of_embeddings(self):
        sim_tokens = [
            "Huhn",
            "Ei",
            "Vogel",
            "Geflügel"
        ]
        diff_tokens = [
            "Code",
            "Geflügel",
            "Siebträger",
            "Donald Trump"
        ]
        sim_ems = self.fe.get_token_embeddings(sim_tokens)
        diff_ems = self.fe.get_token_embeddings(diff_tokens)
        mean_sim = self.fe.mean_of_pairwise_cosine_distances(sim_ems)
        mean_diff = self.fe.mean_of_pairwise_cosine_distances(diff_ems)
        self.assertTrue(mean_sim < mean_diff)