%
% superclass for all inactive normalization options
%
% author: Martin F. Schiffner
% date: 2019-08-10
% modified: 2020-02-17
%
classdef off < regularization.normalizations.normalization

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = off( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers

            %--------------------------------------------------------------
            % 2.) create inactive normalization options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.normalizations.normalization( size );

        end % function objects = off( varargin )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( normalizations_off )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.normalizations.off
            if ~isa( normalizations_off, 'regularization.normalizations.off' )
                errorStruct.message = 'normalizations_off must be regularization.normalizations.off!';
                errorStruct.identifier = 'string:NoNormalizationsOff';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "off"
            strs_out = repmat( "off", size( normalizations_off ) );

        end % function strs_out = string( normalizations_off )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % apply normalization (scalar)
        %------------------------------------------------------------------
        function weighting = apply_scalar( ~, weighting )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class regularization.normalizations.normalization (scalar) for normalization
            % calling function ensures class linear_transforms.weighting (scalar) for weighting

            %--------------------------------------------------------------
            % 2.) apply threshold (scalar)
            %--------------------------------------------------------------
            % do not modify the weighting

        end % function weighting = apply_scalar( ~, weighting )

	end % methods (Access = protected, Hidden)

end % classdef off < regularization.normalizations.normalization
