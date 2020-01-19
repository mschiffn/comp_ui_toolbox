%
% superclass for all direct algorithm options
%
% author: Martin F. Schiffner
% date: 2019-07-30
% modified: 2020-01-18
%
classdef algorithm_direct < scattering.options.algorithm

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = algorithm_direct( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vectors for size
            % superclass ensures positive integers for size

            %--------------------------------------------------------------
            % 2.) create direct algorithm options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options.algorithm( size );

        end % function objects = algorithm_direct( varargin )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( algorithms_direct )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.options.algorithm_direct
            if ~isa( algorithms_direct, 'scattering.options.algorithm_direct' )
                errorStruct.message = 'algorithms_direct must be scattering.options.algorithm_direct!';
                errorStruct.identifier = 'string:NoOptionsAlgorithmDirect';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "direct"
            strs_out = repmat( "direct", size( algorithms_direct ) );

        end % function strs_out = string( algorithms_direct )

	end % methods

end % classdef algorithm_direct < scattering.options.algorithm
