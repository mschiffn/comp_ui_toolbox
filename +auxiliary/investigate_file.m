function varargout = investigate_file( str_filename, str_variables_cell )
% inspect the contents of a *.mat file
%
% author: Martin F. Schiffner
% date: 2019-01-23
% modified: 2019-01-23

    %----------------------------------------------------------------------
	% 1.) initialize local variables
    %----------------------------------------------------------------------
	N_variables_check = numel( str_variables_cell );

    %----------------------------------------------------------------------
	% 2.) initialize return values
    %----------------------------------------------------------------------
	varargout = cell( 1, N_variables_check + 1 );
	for index_check = 1:(N_variables_check + 1)
        varargout{ index_check } = false;
    end

    %----------------------------------------------------------------------
	% 3.) investigate specified file
    %----------------------------------------------------------------------
	if exist( str_filename, 'file' )

        % file exists
        varargout{ 1 } = true;

        % check contents of file
        result_query = whos( '-file', str_filename );
        N_variables_file = numel( result_query );

        for index_check = 1:N_variables_check

            indicator_found = false;
            for index_variable = 1:N_variables_file
                if strcmp( str_variables_cell{index_check}, result_query(index_variable).name )
                    indicator_found = true;
                end
            end

            varargout{index_check + 1} = indicator_found;
        end
	end % if exist(str_filename, 'file')
end