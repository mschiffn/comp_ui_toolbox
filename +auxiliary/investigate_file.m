function varargout = investigate_file( str_filename, varargin )
%
% inspect the contents of a *.mat file
%
% author: Martin F. Schiffner
% date: 2019-01-23
% modified: 2019-05-14
%

    %----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% check number of arguments
	if nargin < 1
        errorStruct.message = 'At least one argument is required!';
        errorStruct.identifier = 'investigate_file:FewArguments';
        error( errorStruct );
    end
    % assertion: nargin >= 1

    %----------------------------------------------------------------------
	% 2.) initialize return values
	%----------------------------------------------------------------------
	varargout = num2cell( false( 1, nargin ) );

    %----------------------------------------------------------------------
	% 3.) investigate specified file
    %----------------------------------------------------------------------
	% check existence of file
	if exist( str_filename, 'file' )

        % file exists
        varargout{ 1 } = true;

        % query contents of file
        result_query = whos( '-file', str_filename );
        N_variables_file = numel( result_query );

        % iterate variables to check
        for index_check = 1:numel( varargin )

            indicator_found = false;

            % iterate variables in file
            for index_variable = 1:N_variables_file
                if strcmp( varargin{ index_check }, result_query( index_variable ).name )
                    indicator_found = true;
                end
            end % for index_variable = 1:N_variables_file

            varargout{ index_check + 1 } = indicator_found;

        end % for index_check = 1:numel( varargin )

	end % if exist( str_filename, 'file' )

end % function varargout = investigate_file( str_filename, varargin )
