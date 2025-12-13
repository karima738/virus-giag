# trainer.py

from pipeline.builder import VirusModelBuilder


def main():
    builder = VirusModelBuilder()

    system = (
        builder
        .load_data("data/patients.csv")
        .preprocess()
        .train_model()
        .evaluate()
        .save()
        .build()
    )

    print("✅ Entraînement terminé")
    print("📊 Métriques :", system.metrics)


if __name__ == "__main__":
    main()
