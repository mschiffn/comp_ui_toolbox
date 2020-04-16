%
% abstract superclass for all wave atom types
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-04-16
%
classdef (Abstract) type

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        pat ( 1, 1 ) char { mustBeMember( pat, { 'p', 'q', 'u' } ), mustBeNonempty } = 'p'      % type of frequency partition which satsifies parabolic scaling relationship
        N_layers ( 1, 1 ) double { mustBeMember( N_layers, [ 1, 2, 4 ] ), mustBeNonempty } = 1	% number of layers in wave atom decomposition

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = type( pat, N_layers )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure valid number of input arguments
            narginchk( 2, 2 );

            % property validation function ensures valid pat
            % property validation function ensures valid N_layers

            % ensure equal number of dimensions and sizes
            [ pat, N_layers ] = auxiliary.ensureEqualSize( pat, N_layers );

            %--------------------------------------------------------------
            % 2.) create wave atom types
            %--------------------------------------------------------------
            % repeat default wave atom type
            objects = repmat( objects, size( pat ) );

            % iterate wave atom types
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).pat = pat( index_object );
                objects( index_object ).N_layers = N_layers( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = type( pat, N_layers )

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
