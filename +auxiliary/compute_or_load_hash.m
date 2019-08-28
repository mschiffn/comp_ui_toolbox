function varargout = compute_or_load_hash( str_format, hdl_function, indices_args_hash, indices_args_pass, varargin )
%
% wrapper for function with handle function_handle and arguments
%
% author: Martin F. Schiffner
% date: 2019-05-16
% modified: 2019-08-23
%

    %----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
    % ensure format string
    if ~ischar( str_format )
        errorStruct.message = 'str_format must be characters!';
        errorStruct.identifier = 'compute_or_load_hash:NoChars';
        error( errorStruct );
    end

	% ensure function handle
	if ~isa( hdl_function, 'function_handle' )
        errorStruct.message = 'hdl_function must be function_handle';
        errorStruct.identifier = 'compute_or_load_hash:NoFunctionHandle';
        error( errorStruct );
    end

    % ensure nonempty indices_args_hash
    if isempty( indices_args_hash )
        indices_args_hash = ( 1:numel( varargin ) );
    end

    % ensure valid indices_args_hash
    mustBePositive( indices_args_hash );
    mustBeInteger( indices_args_hash );
    mustBeLessThanOrEqual( indices_args_hash, numel( varargin ) );

    % ensure nonempty indices_args_pass
    if isempty( indices_args_pass )
        indices_args_pass = ( 1:numel( varargin ) );
    end

    % ensure valid indices_args_pass
    mustBePositive( indices_args_pass );
    mustBeInteger( indices_args_pass );
    mustBeLessThanOrEqual( indices_args_pass, numel( varargin ) );

    %----------------------------------------------------------------------
    % 2.) generate filename based on hashes of arguments
    %----------------------------------------------------------------------
    % number of arguments to hash
    N_args_hash = numel( indices_args_hash );

    % specify cell array for hashes and names
    str_hash_args = cell( 1, N_args_hash );
    str_name_args = cell( 1, N_args_hash );

	% iterate arguments to hash
    for index_hash = 1:N_args_hash

        % compute hash
% TODO: hash collision because hash function ignores parts of the object properties
        str_hash_args{ index_hash } = auxiliary.DataHash( varargin{ indices_args_hash( index_hash ) } );

        % generate name
        str_name_args{ index_hash } = sprintf( 'arg_%d', index_hash );

    end % for index_hash = 1:N_args_hash

	% print hashes into format string
	str_name_file_full = sprintf( str_format, str_hash_args{ : } );

	% get name of directory
	[ str_name_dir, str_name_file, str_name_ext ] = fileparts( str_name_file_full );

    %----------------------------------------------------------------------
    % 3.) check for existence and hash collisions
    %----------------------------------------------------------------------
    % specify cell array for output names
    str_name_out = cell( 1, nargout );

    % iterate outputs
    for index_out = 1:nargout
        str_name_out{ index_out } = sprintf( 'out_%d', index_out );
    end

	% check existence and contents of file
    indicator_exists_out = cell( 1, nargout );
	[ indicator_exists_file, indicator_exists_out{ : } ] = auxiliary.investigate_file( str_name_file_full, str_name_out{ : } );
    indicator_exists_out = cell2mat( indicator_exists_out );

    % check for hash collisions if file exists
	if indicator_exists_file

        % load function arguments to hash from file
        temp = load( str_name_file_full, str_name_args{ : } );

        % iterate arguments to hash
        for index_hash = 1:N_args_hash
% TODO: problematic for function handles!
            % ensure equality of arguments
            if ~isequal( temp.( str_name_args{ index_hash } ), varargin{ indices_args_hash( index_hash ) } )
                errorStruct.message = sprintf( 'Hash collision for argument %d!', indices_args_hash( index_hash ) );
                errorStruct.identifier = 'compute_or_load_hash:HashCollision';
                error( errorStruct );
            end

        end % for index_hash = 1:N_args_hash

	end % if indicator_exists_file

	%----------------------------------------------------------------------
	% 4.) load or compute function values
	%----------------------------------------------------------------------
	if all( indicator_exists_out(:) )

        %------------------------------------------------------------------
        % a) load all function outputs from file
        %------------------------------------------------------------------
        % print status
        time_start = tic;
        str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
        fprintf( '\t %s: loading file %s%s...', str_date_time, str_name_file, str_name_ext );

        % load all function outputs from file
        temp = load( str_name_file_full, str_name_out{ : } );

        % convert structure to cell array
        for index_out = 1:nargout
            varargout{ index_out } = temp.( str_name_out{ index_out } );
        end

        % infer and print elapsed time
        time_elapsed = toc( time_start );
        fprintf( 'done! (%f s)\n', time_elapsed );

    else

        %------------------------------------------------------------------
        % b) execute function
        %------------------------------------------------------------------
        [ varargout{ 1:nargout } ] = hdl_function( varargin{ indices_args_pass } );

        % 
        if indicator_exists_file

            %--------------------------------------------------------------
            % c) append missing function outputs to existing file
            %--------------------------------------------------------------
            % identify missing outputs
            indices_out_missing = find( ~indicator_exists_out );

            % iterate missing outputs
            for index_out = indices_out_missing
                struct_save.( str_name_out{ index_out } ) = varargout{ index_out };
            end

            % append missing outputs to existing file
            save( str_name_file_full, '-struct', 'struct_save', '-append' );

        else

            %--------------------------------------------------------------
            % d) save all arguments to hash and outputs in new file
            %--------------------------------------------------------------
            % iterate arguments to hash
            for index_hash = 1:N_args_hash
                struct_save.( str_name_args{ index_hash } ) = varargin{ indices_args_hash( index_hash ) };
            end

            % iterate outputs
            for index_out = 1:nargout
                struct_save.( str_name_out{ index_out } ) = varargout{ index_out };
            end

            % ensure existence of folder str_name_dir
            [ success, errorStruct.message, errorStruct.identifier ] = mkdir( str_name_dir );
            if ~success
                error( errorStruct );
            end

            % create new file
            save( str_name_file_full, '-struct', 'struct_save', '-v7.3' );

        end % if indicator_exists_file

	end % if all( indicator_exists_out(:) )

end % function varargout = compute_or_load_hash( str_format, hdl_function, varargin )
