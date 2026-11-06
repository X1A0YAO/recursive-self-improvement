import faulthandler, sys
faulthandler.enable()
sys.argv = ["test"]
exec(open("scripts/test_weight_access.py", encoding="utf-8").read().replace('if __name__ == "__main__":\n    main()', "main()"))
