from etl.extract import load_raw_data
from etl.transform import transform_all
from etl.load import load_to_sqlite, test_connection


def main():
    print("=" * 50)
    print("Smart E-Commerce ETL Pipeline")
    print("=" * 50)

    # 1. Extract
    print("\n[1/3] Extract...")
    raw_data = load_raw_data()

    # 2. Transform
    print("\n[2/3] Transform...")
    transformed = transform_all(raw_data)

    # 3. Load
    print("\n[3/3] Load...")
    load_to_sqlite(transformed)

    # Test
    print("\n[CHECK] Database test:")
    test_connection()

    print("\n[SUCCESS] ETL pipeline completed!")


if __name__ == "__main__":
    main()