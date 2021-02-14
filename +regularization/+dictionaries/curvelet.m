%
% superclass for all curvelet dictionaries
%
% author: Martin F. Schiffner
% date: 2020-10-27
% modified: 2020-11-11
%
classdef curvelet < regularization.dictionaries.dictionary

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = curvelet( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin < 1 || isempty( size )
                size = 1;
            end

            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create curvelet dictionaries
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.dictionaries.dictionary( size );

        end % function objects = curvelet( size )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( dictionaries_curvelet )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.dictionaries.curvelet
            if ~isa( dictionaries_curvelet, 'regularization.dictionaries.curvelet' )
                errorStruct.message = 'dictionaries_curvelet must be regularization.dictionaries.curvelet!';
                errorStruct.identifier = 'string:NoDictionaryCurvelet';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "curvelet"
            strs_out = repmat( "curvelet", size( dictionaries_curvelet ) );

        end % function strs_out = string( dictionaries_curvelet )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % create linear transform (scalar)
        %------------------------------------------------------------------
        function LT = get_LT_scalar( ~, operator_born )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class regularization.dictionaries.dictionary (scalar)
            % calling function ensures class scattering.operator_born (scalar)

            %--------------------------------------------------------------
            % 2.) create linear transform (scalar)
            %--------------------------------------------------------------
            % extract non-singleton dimensions
            indicator_dimensions = operator_born.sequence.setup.FOV.shape.grid.N_points_axis > 1;

            % create linear transform
            LT = linear_transforms.curvelet( operator_born.sequence.setup.FOV.shape.grid.N_points_axis( indicator_dimensions ) );

        end % function LT = get_LT_scalar( ~, operator_born )

	end % methods (Access = protected, Hidden)

end % classdef curvelet < regularization.dictionaries.dictionary
