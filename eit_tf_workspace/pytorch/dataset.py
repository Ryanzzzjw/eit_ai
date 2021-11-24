
if __name__ == "__main__":
    from glob_utils.log.log  import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)