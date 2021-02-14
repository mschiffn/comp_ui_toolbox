%
% abstract superclass for all wave atom types
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-10-31
%
classdef (Abstract) type

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        pat ( 1, 1 ) char { mustBeMember( pat, { 'p', 'q', 'u' } ), mustBeNonempty } = 'p' % type of frequency partition which satsifies parabolic scaling relationship

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = type( pat )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty pat
            if nargin < 1 || isempty( pat )
                pat = 'p';
            end

            % property validation function ensures valid pat

            %--------------------------------------------------------------
            % 2.) create wave atom types
            %--------------------------------------------------------------
            % repeat default wave atom type
            objects = repmat( objects, size( pat ) );

            % iterate wave atom types
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).pat = pat( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = type( pat )

        %------------------------------------------------------------------
        % parameters for function call
        %------------------------------------------------------------------
        function params = get_parameters( types )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure one argument
            narginchk( 1, 1 );

            % ensure class linear_transforms.wave_atoms.type
            if ~isa( types, 'linear_transforms.wave_atoms.type' )
                errorStruct.message = 'types must be linear_transforms.wave_atoms.type!';
                errorStruct.identifier = 'get_parameters:NoWaveAtomTypes';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) return parameters for function call
            %--------------------------------------------------------------
            % specify cell array
            params = cell( size( types ) );

            % iterate wave atom types
            for index_object = 1:numel( types )

                % create cell array w/ parameters for function call
                params{ index_object } = get_parameters_scalar( types( index_object ) );

            end % for index_object = 1:numel( types )

            % avoid cell array for single types
            if isscalar( types )
                params = params{ 1 };
            end

        end % function params = get_parameters( types )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract)

        %------------------------------------------------------------------
        % numbers of layers
        %------------------------------------------------------------------
        N_layers = get_N_layers( types, N_dimensions )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        strs_out = string( types )

	end % methods (Abstract)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract, protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % parameters for function call (scalar)
        %------------------------------------------------------------------
        params = get_parameters_scalar( type )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) type
