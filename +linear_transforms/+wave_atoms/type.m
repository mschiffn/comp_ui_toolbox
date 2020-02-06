%
% abstract superclass for all wave atom types
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-01-30
%
classdef (Abstract) type

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_layers ( 1, 1 ) double { ismember( N_layers, [ 1, 2, 4 ] ) } = 1

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = type( N_layers )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation function ensures valid N_layers

            %--------------------------------------------------------------
            % 2.) create wave atom types
            %--------------------------------------------------------------
            % repeat default wave atom type
            objects = repmat( objects, size( N_layers ) );

            % iterate wave atom types
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).N_layers = N_layers( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = type( N_layers )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract)

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        strs_out = string( types )

	end % methods (Abstract)

end % classdef (Abstract) type
