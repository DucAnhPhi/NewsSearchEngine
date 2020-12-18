import unittest
import os
from ...scrapy.http import HtmlResponse, Request
from ...netzpolitik.scraper import NetzpolitikScraper
from ...netzpolitik.parser import ParserNetzpolitik

def fake_response_from_file(file_name, url=None):
    """
    Create a Scrapy fake HTTP response from a HTML file
    @param file_name: The relative filename from the responses directory,
                      but absolute paths are also accepted.
    @param url: The URL of the response.
    returns: A scrapy HTTP response which can be used for unittesting.
    """
    if not url:
        url = 'http://www.example.com'

    request = Request(url=url)
    if not file_name[0] == '/':
        responses_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(responses_dir, file_name)
    else:
        file_path = file_name

    with open(file_path, 'r') as f:
        file_content = f.read()

    response = HtmlResponse(url=url,
                            request=request,
                            body=file_content,
                            encoding='utf-8')
    return response


class TestParserNetzpolitik(unittest.TestCase):
    
    def test_parse_2020(self):
        fake_response = fake_response_from_file(
            'html/netzpolitik_2020.html', url='https://netzpolitik.org/2020/eu-rechnungshof-kartellbehoerden-sollen-tech-konzerne-haerter-anfassen/')
        parsed = next(ParserNetzpolitik.parse_article(fake_response))
        self.assertEqual(
            parsed["title"], "Kartellbehörden sollen Tech-Konzerne härter anfassen")
        self.assertEqual(parsed["subtitle"], "EU-Rechnungshof")
        self.assertTrue(set(parsed["authors"]) == {"Serafin Dinges"})
        self.assertEqual(parsed["published"], "19-11-2020")
        self.assertTrue(set(parsed["categories"]) == {"Netzpolitik"})
        keywords = {
            "Amazon",
            "Apple",
            "eu-kommission",
            "EU-Wettbewerbskommission",
            "Europäische Union",
            "facebook",
            "google",
            "Kartell",
            "Kartellrecht",
            "Sonderbericht",
            "Wettbewerb"
        }
        self.assertTrue(set(parsed["keywords"]) == keywords)
        refs = {
            "https://www.eca.europa.eu/de/Pages/DocItem.aspx?did=56835",
            "https://netzpolitik.org/2017/eu-kommission-verdonnert-google-zu-24-milliarden-euro-strafe/",
            "https://netzpolitik.org/2020/datenschutz-studie-privatsphaere-und-wettbewerb-zusammendenken/",
            "https://netzpolitik.org/2020/bundesgerichtshof-facebook-beutet-nutzer-kartellrechtlich-relevant-aus/",
            "https://netzpolitik.org/2020/eu-plattformgrundgesetz-digital-services-act/",
            "https://ec.europa.eu/transparency/regdoc/rep/2/2020/EN/SEC-2020-2357-F1-EN-MAIN-PART-1.PDF",
            "https://netzpolitik.org/2020/wettbewerbsrecht-eu-kommission-prueft-amazons-datenmacht/",
            "https://netzpolitik.org/2020/us-ausschuss-zu-tech-monopolisten-bis-zur-zerschlagung/",
            "https://www.spiegel.de/netzwelt/apps/apple-senkt-abgabe-fuer-kleine-app-entwickler-a-8136c92c-9171-4f0b-8717-a7ff8431ef94"
        }
        self.assertTrue(set(parsed["references"]) == refs or refs.issubset(set(parsed["references"])))
        expected_section_titles = [
            "Den Markt neu definieren",
            "Neue Regulierungen stehen in Aussicht"
        ]
        actual_section_titles = ParserNetzpolitik.get_section_titles(parsed)
        self.assertTrue(len(expected_section_titles) == len(actual_section_titles))
        self.assertTrue(set(expected_section_titles) == set(actual_section_titles))
        expected_text = "EU-Rechnungshof Kartellbehörden sollen Tech-Konzerne härter anfassen Facebook, Amazon, Google und Apple konnten jahrelang ungehindert ihren Marktvorteil ausbauen. Wenn reguliert wurde, dann nur träge. Ein Sonderbericht des EU-Rechnungshofs zieht ein kritisches Fazit."
        actual_text = ParserNetzpolitik.get_first_paragraph_with_titles(parsed)
        self.assertEqual(actual_text, expected_text)

    def test_parse_2019(self):
        fake_response = fake_response_from_file(
            'html/netzpolitik_2019.html', url='https://netzpolitik.org/2019/npp191-off-the-record-die-tiktok-recherche-und-ein-neues-gesicht/')
        parsed = next(ParserNetzpolitik.parse_article(fake_response))
        self.assertEqual(
            parsed["title"], "Die TikTok-Recherche und ein neues Gesicht")
        self.assertEqual(parsed["subtitle"], "NPP 191 Off The Record")
        self.assertTrue(set(parsed["authors"]), {"Markus Reuter", "Chris Köver"})
        self.assertEqual(parsed["published"], "07-12-2019")
        self.assertTrue(set(parsed["categories"]) == {"Netzpolitik Podcast"})
        self.assertTrue(set(parsed["keywords"]) == {"Netzpolitik-Podcast", "NPP Off The Record"})
        refs = {
            "https://netzpolitik.org/moderation-tiktok-informationskontrolle/",
            "https://netzpolitik.org/2019/gute-laune-und-zensur/",
            "https://netzpolitik.org/2019/die-kritik-drossel-von-tiktok/",
            "https://netzpolitik.org/2019/tiktoks-obergrenze-fuer-behinderungen/",
            "https://netzpolitik.org/2019/wie-8chan-unter-neuem-namen-zurueckkehren-soll/",
            "https://netzpolitik.org/2019/8chan-8kun-das-imageboard-hat-die-falschen-freunde/",
            "https://www.zdf.de/politik/frontal-21/neue-spuren-vom-halle-attentaeter-100.html",
            "https://www.washingtonpost.com/technology/2019/09/15/tiktoks-beijing-roots-fuel-censorship-suspicion-it-builds-huge-us-audience/",
            "https://www.theguardian.com/technology/2019/sep/25/revealed-how-tiktok-censors-videos-that-do-not-please-beijing",
            "https://netzpolitik.org/podcast/",
            "https://cdn.netzpolitik.org/wp-upload/2019/12/npp191-offtherecord-tiktok.ogg",
            "https://cdn.netzpolitik.org/wp-upload/2019/12/npp191-offtherecord-tiktok.mp3",
            "spotify:show:2GLuMhSNEFzUIXfx9BDxBt"
        }
        self.assertTrue(set(parsed["references"]) == refs or refs.issubset(set(parsed["references"])))
        expected_section_titles = [
            "Shownotes"
        ]
        actual_section_titles = ParserNetzpolitik.get_section_titles(parsed)
        self.assertTrue(len(expected_section_titles) == len(actual_section_titles))
        self.assertTrue(set(expected_section_titles) == set(actual_section_titles))
        expected_text = "NPP 191 Off The Record Die TikTok-Recherche und ein neues Gesicht Lustige Videos, heile Welt. Die Videoplattform TikTok ist das am schnellsten wachsende soziale Netzwerk. Wir haben seit August über Moderation und Inhaltskontrolle auf der chinesischen Plattform recherchiert – und geben im Podcast nun einen Blick hinter die Kulissen."
        actual_text = ParserNetzpolitik.get_first_paragraph_with_titles(parsed)
        self.assertEqual(actual_text, expected_text)

    def test_parse_2018(self):
        fake_response = fake_response_from_file(
            'html/netzpolitik_2018.html', url='https://netzpolitik.org/2018/die-it-tools-des-bamf-fehler-vorprogrammiert/')
        parsed = next(ParserNetzpolitik.parse_article(fake_response))
        self.assertEqual(
            parsed["title"], "Die IT-Tools des BAMF: Fehler vorprogrammiert")
        self.assertEqual(parsed["subtitle"], None)
        self.assertTrue(set(parsed["authors"]), {"Anna Biselli"})
        self.assertEqual(parsed["published"], "28-12-2018")
        self.assertTrue(set(parsed["categories"]) == {"Öffentlichkeit"})
        keywords = {
            "Asyl",
            "asylverfahren",
            "BAMF",
            "dialektanalyse",
            "Geflüchtete",
            "handyauswertung",
            "IFG",
            "Informationsfreiheit",
            "it",
            "Kleine Anfrage",
            "künstliche intelligenz",
            "sprachanalyse",
            "Sprachbiometrie",
            "transliteration"
        }
        self.assertTrue(set(parsed["keywords"]) == keywords)
        refs = {
            "https://netzpolitik.org/2018/asylverfahren-handy-durchsuchung-bringt-keine-vorteile/",
            "https://netzpolitik.org/2018/das-bamf-will-seine-probleme-mit-technik-loesen-und-macht-alles-noch-schlimmer/",
            "https://netzpolitik.org/2017/bundesamt-fuer-migration-und-fluechtlinge-rueckt-gefluechteten-mit-neuer-software-auf-die-pelle/",
            "https://www.vhsit.berlin.de/VHSKURSE/BusinessPages/CourseDetail.aspx?id=506356",
            "https://netzpolitik.org/2017/syrien-oder-aegypten-software-zur-dialektanalyse-ist-fehleranfaellig-und-intransparent/",
            "https://fragdenstaat.de/anfrage/foliensatze-und-interpretationshilfen-zu-sprachanalyse/",
            "https://cdn.netzpolitik.org/wp-upload/2018/12/schulung_idms_bamf.pdf",
            "https://fragdenstaat.de/anfrage/foliensatze-und-interpretationshilfen-zu-sprachanalyse/",
            "https://fragdenstaat.de/anfrage/dienstanweisungen-zum-umgang-mit-der-handyauswertung/",
            "https://www.bamf.de/SharedDocs/Meldungen/DE/2018/20181205-am-digitalgipfel.html",
            "https://motherboard.vice.com/de/article/a3q8wj/fluechtlinge-bamf-sprachanalyse-software-entscheidet-asyl",
            "https://www.wr.de/politik/bamf-nach-ueberforderung-bei-fluechtlingskrise-neu-aufgestellt-id216038109.html",
            "https://motherboard.vice.com/de/article/kzv5v3/sprachanalyse-handyauswertung-bamf-it-fluechtlinge-herkunft",
        }
        self.assertTrue(set(parsed["references"]) == refs or refs.issubset(set(parsed["references"])))
        expected_section_titles = [
            "Software und Wahrscheinlichkeiten versprechen trügerische Sicherheit",
            "Bedienungs- und Interpretationsfehler sind vorprogrammiert",
            "Handyauswertungen sind nur in 35 Prozent der Fälle überhaupt verwertbar"
        ]
        actual_section_titles = ParserNetzpolitik.get_section_titles(parsed)
        self.assertTrue(len(expected_section_titles) == len(actual_section_titles))
        self.assertTrue(set(expected_section_titles) == set(actual_section_titles))
        expected_text = "Die IT-Tools des BAMF: Fehler vorprogrammiert Das Bundesamt für Migration und Flüchtlinge (BAMF) will mit Auswertungen von Smartphones sowie Namens- und Dialektanalysen herausfinden, woher Geflüchtete kommen. Die Schulungen, die BAMF-Mitarbeiter dazu durchlaufen, geben ihnen jedoch kaum Anhaltspunkte, wie sie die Ergebnisse ihrer digitalen Untersuchungen interpretieren sollen. Wir veröffentlichen die Dokumente."
        actual_text = ParserNetzpolitik.get_first_paragraph_with_titles(parsed)
        self.assertEqual(actual_text, expected_text)

    def test_parse_2017(self):
        fake_response = fake_response_from_file(
            'html/netzpolitik_2017.html', url='https://netzpolitik.org/2017/ein-text-den-ich-noch-nicht-lesen-kann-erlebnisbericht-eines-neulings-auf-dem-chaos-communication-congress/')
        parsed = next(ParserNetzpolitik.parse_article(fake_response))
        self.assertEqual(
            parsed["title"], "Ein Text, den ich noch nicht lesen kann. Erlebnisbericht eines Neulings auf dem Chaos Communication Congress")
        self.assertEqual(parsed["subtitle"], None)
        self.assertTrue(set(parsed["authors"]), {"Stefanie Talaska"})
        self.assertEqual(parsed["published"], "27-12-2017")
        self.assertTrue(set(parsed["categories"]) == {"Kultur"})
        keywords = {
            "34c3",
            "Chaos Communication Congress",
            "Chaos Computer Club",
            "Constanze Kurz",
            "Erlebnisbericht",
            "Hans-Christian Ströbele",
            "Leipzig",
            "NSAUA",
            "wau_holland"
        }
        self.assertTrue(set(parsed["keywords"]) == keywords)
        refs = {
            "https://lns.wtf/twt/Design_Guide/171129_34C3-Design_Public.pdf",
            "https://media.ccc.de/v/34c3-9292-eroffnung_tuwat",
            "https://media.ccc.de/v/34c3-9289-die_lauschprogramme_der_geheimdienste",
            "https://media.ccc.de/v/34c3-9092-ladeinfrastruktur_fur_elektroautos_ausbau_statt_sicherheit",
            "http://luftdaten.info/",
            "https://de.wikipedia.org/wiki/Das_Schwarze_Quadrat"
        }
        self.assertTrue(set(parsed["references"]) == refs or refs.issubset(set(parsed["references"])))
        expected_section_titles = [
            "Alles ist Text, der verstanden werden will.",
            "„Es würde mir schon reichen, einfach Hardware anzufassen.“"
        ]
        actual_section_titles = ParserNetzpolitik.get_section_titles(parsed)
        self.assertTrue(len(expected_section_titles) == len(actual_section_titles))
        self.assertTrue(set(expected_section_titles) == set(actual_section_titles))
        expected_text = "Ein Text, den ich noch nicht lesen kann. Erlebnisbericht eines Neulings auf dem Chaos Communication Congress Teil II Was ist das? Was sagt er da? Was machen die da? Der Congress-Neuling schnappt Dinge auf, hört zu und beobachtet. Es ergibt sich ein Text, der nicht all zu leicht zu verstehen ist."
        actual_text = ParserNetzpolitik.get_first_paragraph_with_titles(parsed)
        self.assertEqual(actual_text, expected_text)

    def test_parse_2016(self):
        fake_response = fake_response_from_file(
            'html/netzpolitik_2016.html', url='https://netzpolitik.org/2016/interview-kampf-der-abmahnindustrie/')
        parsed = next(ParserNetzpolitik.parse_article(fake_response))
        self.assertEqual(
            parsed["title"], "Interview: Kampf der Abmahnindustrie")
        self.assertEqual(parsed["subtitle"], None)
        self.assertTrue(set(parsed["authors"]), {"Ingo Dachwitz"})
        self.assertEqual(parsed["published"], "30-12-2016")
        self.assertTrue(set(parsed["categories"]) == {"Netze"})
        keywords = {
            "33C3",
            "Abmahnanwalt",
            "abmahnbeantworter",
            "Abmahnindustrie",
            "Beata Hubrig",
            "CCC",
            "Chaos Communication Congress",
            "Chaos Computer Club",
            "Erdgeist",
            "freifunk",
            "Offenes Netz",
            "Störerhaftung",
            "Urheberrecht",
            "WLAN-Störerhaftung"
        }
        self.assertTrue(set(parsed["keywords"]) == keywords)
        refs = {
            "https://netzpolitik.org/2016/abmahnbeantworter-neues-tool-hilft-unberechtigt-abgemahnten-bei-gegenwehr/",
            "https://netzpolitik.org/2016/nach-dem-eugh-urteil-eine-abschaffung-der-stoererhaftung-ist-trotzdem-moeglich/",
            "https://abmahnbeantworter.ccc.de/",
            "https://twitter.com/lolaandromeda",
            "https://twitter.com/erdgeist",
            "https://media.ccc.de/v/33c3-8388-kampf_dem_abmahnunwesen"
        }
        self.assertTrue(set(parsed["references"]) == refs or refs.issubset(set(parsed["references"])))
        expected_section_titles = [
            "Geschäft mit der Angst",
            "Den Spieß umdrehen: Kostenrisiko für Abmahner",
            "Hoffnung auf Rechtssicherheit für offene Netze"
        ]
        actual_section_titles = ParserNetzpolitik.get_section_titles(parsed)
        self.assertTrue(len(expected_section_titles) == len(actual_section_titles))
        self.assertTrue(set(expected_section_titles) == set(actual_section_titles))
        expected_text = "Interview: Kampf der Abmahnindustrie Nach wie vor verdienen Anwälte mit automatisierten Urheberrechtsabmahnungen gutes Geld. Dabei bedienen sie sich unsauberer Methoden, sagen die Initiatoren des „Abmahnbeantworters“. Auf dem 33C3 schlagen sie vor, den Spieß umzudrehen und die Kanzleien hinter den Massenabmahnungen selbst zur Kasse zu bitten."
        actual_text = ParserNetzpolitik.get_first_paragraph_with_titles(parsed)
        self.assertEqual(actual_text, expected_text)

    def test_parse_2015(self):
        fake_response = fake_response_from_file(
            'html/netzpolitik_2015.html', url='https://netzpolitik.org/2015/32c3-ein-abgrund-von-landesverrat/')
        parsed = next(ParserNetzpolitik.parse_article(fake_response))
        self.assertEqual(
            parsed["title"], "#32c3: Ein Abgrund von #Landesverrat")
        self.assertEqual(parsed["subtitle"], None)
        self.assertTrue(set(parsed["authors"]), {"Markus Beckedahl"})
        self.assertEqual(parsed["published"], "31-12-2015")
        self.assertTrue(set(parsed["categories"]) == {"Linkschleuder"})
        keywords = {
            "32c3",
            "CCC",
            "Chaos Communication Congress",
            "Kurzmeldungen",
            "Landesverrat",
            "Netzpolitik"
        }
        self.assertTrue(set(parsed["keywords"]) == keywords)
        refs = {
            "https://media.ccc.de/v/32c3-7135-ein_abgrund_von_landesverrat",
            "https://media.ccc.de/v/32c3-7135-ein_abgrund_von_landesverrat",
            "https://www.youtube.com/watch?v=dBmIbrPQpQY",
            "http://cdn.media.ccc.de/congress/2015/h264-hd/32c3-7135-de-en-Ein_Abgrund_von_Landesverrat_hd.mp4"
        }
        self.assertTrue(set(parsed["references"]) == refs or refs.issubset(set(parsed["references"])))
        expected_section_titles = []
        actual_section_titles = ParserNetzpolitik.get_section_titles(parsed)
        self.assertTrue(len(expected_section_titles) == len(actual_section_titles))
        expected_text = "#32c3: Ein Abgrund von #Landesverrat Auf dem 32. Chaos Communication Congress hab ich nochmal die Ermittlungen wegen Landesverrats gegen uns zusammengefasst und auch ein kleines Fazit gezogen. Die halbe Stunde Vortrag findet sich hier in der CCC-Mediathek und auf Youtube."
        actual_text = ParserNetzpolitik.get_first_paragraph_with_titles(parsed)
        self.assertEqual(actual_text, expected_text)

    def test_parse_2014(self):
        fake_response = fake_response_from_file(
            'html/netzpolitik_2014.html', url='https://netzpolitik.org/2014/netzpolitischer-jahresrueckblick-2014-dezember/')
        parsed = next(ParserNetzpolitik.parse_article(fake_response))
        self.assertEqual(
            parsed["title"], "Netzpolitischer Jahresrückblick 2014: Dezember")
        self.assertEqual(parsed["subtitle"], None)
        self.assertTrue(set(parsed["authors"]), {"Anna Biselli"})
        self.assertEqual(parsed["published"], "31-12-2014")
        self.assertTrue(set(parsed["categories"]) == {"Generell"})
        self.assertTrue(set(parsed["keywords"]) == {"2014", "Aus der Reihe", "jahresrückblick"})
        refs = {
            "https://netzpolitik.org/2014/netzpolitischer-jahresrueckblick-2014-januar/",
            "https://netzpolitik.org/2014/netzpolitischer-jahresrueckblick-2014-februar/",
            "https://netzpolitik.org/2014/netzpolitischer-jahresrueckblick-2014-maerz/",
            "https://netzpolitik.org/2014/netzpolitischer-jahresrueckblick-2014-april/",
            "https://netzpolitik.org/2014/netzpolitischer-jahresrueckblick-2014-mai/",
            "https://netzpolitik.org/2014/netzpolitischer-jahresrueckblick-2014-juni/",
            "https://netzpolitik.org/2014/netzpolitischer-jahresrueckblick-2014-juli/",
            "https://netzpolitik.org/2014/netzpolitischer-jahresrueckblick-2014-august/",
            "https://netzpolitik.org/2014/netzpolitischer-jahresrueckblick-2014-september/",
            "https://netzpolitik.org/2014/netzpolitischer-jahresrueckblick-2014-oktober/",
            "https://netzpolitik.org/2014/netzpolitischer-jahresrueckblick-2014-november/",
            "https://netzpolitik.org/2014/einstieg-ins-zweiklassennetz-bundesregierung-legt-gemeinsames-position-zur-netzneuralitraet-vor/",
            "https://netzpolitik.org/2014/schon-70-000-mitzeichner-fuer-petition-netzneutralitaet-sichern-rettet-das-freie-internet/",
            "https://netzpolitik.org/2014/anlasslose-massenueberwachung-eu-kommission-arbeitet-an-neuer-richtlinie-zur-vorratsdatenspeicherung/",
            "https://netzpolitik.org/2014/unbelehrbar-cdu-fordert-vorratsdatenspeicherung-und-quellen-tkue/",
            "https://netzpolitik.org/2014/nsa-ausschuss-bundesregierung-will-snowden-dokument-nicht-herausgeben/",
            "https://netzpolitik.org/2014/bundesverfassungsgericht-klage-von-linken-und-gruenen-zur-vernehmung-von-snowden-in-berlin-unzulaessig/",
            "https://netzpolitik.org/2014/nach-razzia-in-schweden-the-pirate-bay-ist-offline/",
            "https://netzpolitik.org/2014/nach-razzia-the-pirate-bay-ist-unter-neuer-domain-wieder-da/",
            "https://netzpolitik.org/2014/peter-sunde-ich-bin-fuer-meine-sache-ins-gefaengnis-gegangen-was-hast-du-gemacht/",
            "https://netzpolitik.org/2014/viewable-everywhere-except-germany-mehr-transparenz-bei-youtubes-content-id-verfahren/",
            "https://netzpolitik.org/2014/live-blog-der-28-sitzung-nsa-untersuchungsausschuss/",
            "https://netzpolitik.org/2014/obmann-der-cducsu-ueber-nsaua-bisher-nicht-ein-einziger-hinweis-auf-anlasslose-massenueberwachung/",
            "https://netzpolitik.org/2014/it-sicherheitsgesetz-im-kabinett-beschlossen-die-kritischen-punkte-zusammengefasst/",
            "https://netzpolitik.org/2014/pressekonferenz-zum-it-sicherheitsgesetz-mit-anonymer-meldepflicht-gegen-premiumangriffe/",
            "https://netzpolitik.org/2014/vupen-threat-protection-wir-veroeffentlichen-den-vertrag-mit-dem-das-bsi-sicherheitsluecken-und-exploits-kauft/",
            "https://netzpolitik.org/2014/informationsfreiheits-ablehnung-des-tages-information-ueber-spionage-firmen-koennte-us-botschaft-nachhaltig-stoeren/",
            "https://netzpolitik.org/2014/leak-zeigt-handelsabkommen-tisa-koennte-nationale-datenschutzbestimmungen-aushebeln/"
        }
        self.assertTrue(set(parsed["references"]) == refs or refs.issubset(set(parsed["references"])))
        expected_section_titles = []
        actual_section_titles = ParserNetzpolitik.get_section_titles(parsed)
        self.assertTrue(len(expected_section_titles) == len(actual_section_titles))
        expected_text = "Netzpolitischer Jahresrückblick 2014: Dezember Heute endet das Jahr 2014 und damit auch unser Jahresrückblick. In den letzten zwei Wochen haben wir jeden Tag auf je einen Monat des Jahres zurückgeblickt und geschaut, was im und um das Netz wichtig war."
        actual_text = ParserNetzpolitik.get_first_paragraph_with_titles(parsed)
        self.assertEqual(actual_text, expected_text)

    def test_parse_2013(self):
        fake_response = fake_response_from_file(
            'html/netzpolitik_2013.html', url='https://netzpolitik.org/2013/vorsatz-fuer-2014-mithelfen-das-urheberrecht-zu-modernisieren/')
        parsed = next(ParserNetzpolitik.parse_article(fake_response))
        self.assertEqual(
            parsed["title"], "Vorsatz für 2014: Mithelfen das Urheberrecht zu modernisieren!")
        self.assertEqual(parsed["subtitle"], None)
        self.assertTrue(set(parsed["authors"]), {"Leonhard Dobusch"})
        self.assertEqual(parsed["published"], "30-12-2013")
        self.assertTrue(set(parsed["categories"]) == {"Wissen"})
        keywords = {
            "30C3",
            "EU-Konsultation",
            "Help Reform Copyright",
            "OKFN",
            "rechtaufremix.org",
            "Stefan Wehrmeyer",
            "Urheberrecht",
            "Urheberrechtsreform"
        }
        self.assertTrue(set(parsed["keywords"]) == keywords)
        refs = {
            "http://okfde.github.io/eucopyright/",
            "http://okfde.github.io/eucopyright/en/30c3/",
            "http://events.ccc.de/congress/2013/Fahrplan/events/5433.html"
        }
        self.assertTrue(set(parsed["references"]) == refs or refs.issubset(set(parsed["references"])))
        expected_section_titles = []
        actual_section_titles = ParserNetzpolitik.get_section_titles(parsed)
        self.assertTrue(len(expected_section_titles) == len(actual_section_titles))
        expected_text = "Vorsatz für 2014: Mithelfen das Urheberrecht zu modernisieren! Das netzpolitische Jahr 2014 beginnt auf europäischer Ebene mit einer öffentlichen Konsultation zur Evaluierung der EU-Urheberrechtsrichtline. Und kaum ein Rechtsbestand ist derart überarbeitungsbedürftig wie das europäische Urheberrecht. Gleichzeitig ist die Teilnahme an so einer Konsultation gerade für Laien oft schwierig, weil die Fragenkataloge lang und unübersichtlich sind – und das, obwohl gerade der Input von „normalen“ InternetnutzerInnen besonders gefragt ist."
        actual_text = ParserNetzpolitik.get_first_paragraph_with_titles(parsed)
        self.assertEqual(actual_text, expected_text)

    def test_parse_2012(self):
        fake_response = fake_response_from_file(
            'html/netzpolitik_2012.html', url='https://netzpolitik.org/2012/raubkopierer-sind-verbrecher-im-zdf-schleichwerbung-bei-soko-stuttgart/')
        parsed = next(ParserNetzpolitik.parse_article(fake_response))
        self.assertEqual(
            parsed["title"], "Raubkopierer sind Verbrecher im ZDF: Schleichwerbung bei SOKO Stuttgart?")
        self.assertEqual(parsed["subtitle"], None)
        self.assertTrue(set(parsed["authors"]), {"Leonhard Dobusch"})
        self.assertEqual(parsed["published"], "31-12-2012")
        self.assertTrue(set(parsed["categories"]) == {"Wissen"})
        keywords = {
            "filmindustrie",
            "hartabergerecht",
            "Raubkopierer sind Verbrecher",
            "Schleichwerbung",
            "SOKO Stuttgart",
            "Urheberrecht",
            "ZDF"
        }
        self.assertTrue(set(parsed["keywords"]) == keywords)
        refs = {
            "https://netzpolitik.org/2011/deutsche-content-allianz/",
            "https://netzpolitik.org/2005/ard-schleichwerbung-fur-die-gvu/",
            "http://sokostuttgart.zdf.de/ZDF/zdfportal/programdata/d4a31cf8-a4fe-3ac7-9cfa-185b43d1216d/20103488",
            "http://de.wikipedia.org/wiki/Raubkopierer_sind_Verbrecher",
            "http://www.respectcopyrights.de/",
            "http://www.spiegel.de/spiegel/print/d-84339496.html",
            "http://www.raubkopierer-sind-verbrecher.de/raubkopierer-sind-verbrecher.htm",
            "http://www.youtube.com/watch?v=VcXcClHA750"
        }
        self.assertTrue(set(parsed["references"]) == refs or refs.issubset(set(parsed["references"])))
        expected_section_titles = []
        actual_section_titles = ParserNetzpolitik.get_section_titles(parsed)
        self.assertTrue(len(expected_section_titles) == len(actual_section_titles))
        expected_text = "Raubkopierer sind Verbrecher im ZDF: Schleichwerbung bei SOKO Stuttgart? Der öffentlich-rechtliche Rundfunk in Deutschland bekleckert sich als Mitglied der „Deutschen Content Allianz“ nicht gerade mit Ruhm wenn es um eine zeitgemäße Reform des Urheberrechts geht und die ARD bewegte sich auch in diesem Themenfeld bereits hart an der Grenze zur Schleichwerbung."
        actual_text = ParserNetzpolitik.get_first_paragraph_with_titles(parsed)
        self.assertEqual(actual_text, expected_text)